import functools
from loguru import logger
import os
import time
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, cast, Dict

import tyro
import jax

# jax.distributed.initialize()

import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from chex import PRNGKey
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from flax import nnx
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import DiTModel
from sampling import rectified_flow_sample, rectified_flow_step
from utils import (
    center_crop,
    ensure_directory,
    image_grid,
    normalize_images,
    process_raw_dict,
)
from profiling import memory_usage_params, trace_module_calls, get_peak_flops
from jax import Array

from vae.vae_flax import load_pretrained_vae
from options import AllConfigs, Config


jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")

if jax.process_index() == 0:
    logger.info(f"JAX host count: {jax.process_count()}")
    logger.info(f"JAX device count: {jax.device_count()}")


def fmt_float_display(val: Array | float | int) -> str:
    if val > 1e3:
        return f"{val:.2e}"
    return f"{val:3.3f}"


class Trainer:

    def __init__(
        self,
        rng: PRNGKey,
        config: Config,
        learning_rate: float = 5e-5,
        profile: bool = False,
    ) -> None:

        init_key, self.train_key = random.split(rng, 2)
        latent_size, n_channels = config.latent_size, config.n_channels
        dtype = jnp.bfloat16 if config.half_precision else jnp.float32

        @nnx.jit
        def create_sharded_model():
            model = DiTModel(
                dim=config.model_config.dim,
                n_layers=config.model_config.n_layers,
                n_heads=config.model_config.n_heads,
                input_size=latent_size,
                in_channels=n_channels,
                out_channels=n_channels,
                n_classes=config.n_classes,
                dtype=dtype,
                rngs=nnx.Rngs(init_key),
            )
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model

        n_devices = len(jax.devices())

        # Create a device mesh according to the physical layout of the devices.
        # device_mesh is just an ndarray
        device_mesh = mesh_utils.create_device_mesh((n_devices, 1))
        if jax.process_index() == 0:
            logger.info(f"Available devices: {jax.devices()}")
            logger.info(f"Device mesh: {device_mesh}")

        # The axes are (data, model), so the mesh is (n_devices, 1) as the model is replicated across devices.
        # This object corresponds the axis names to the layout of the physical devices,
        # so that sharding a tensor along the axes shards according to the corresponding device_mesh layout.
        # i.e. with device layout of (8, 1), data would be replicated to all devices, and model would be replicated to 1 device.
        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))
        if jax.process_index() == 0:
            logger.info(f"Mesh: {self.mesh}")
            logger.info(f"Initializing model...")

        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("data"))
        self.key_sharding = NamedSharding(self.mesh, PartitionSpec())

        with self.mesh:
            self.model = create_sharded_model()

        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            best_mode="min",
            multiprocessing_options=ocp.options.MultiprocessingOptions(
                primary_host=None
            ),
        )

        if config.resume:
            ckpt_path = os.path.join(os.getcwd(), "checkpoints")
        else:
            ckpt_path = ocp.test_utils.erase_and_create_empty(
                os.path.join(os.getcwd(), "checkpoints")
            )

        self.checkpointer = ocp.CheckpointManager(
            ckpt_path,
            options=ckpt_options,
        )

        if config.resume:
            abs_model = nnx.eval_shape(
                lambda: DiTModel(
                    dim=config.model_config.dim,
                    n_layers=config.model_config.n_layers,
                    n_heads=config.model_config.n_heads,
                    input_size=latent_size,
                    in_channels=n_channels,
                    out_channels=n_channels,
                    n_classes=config.n_classes,
                    dtype=dtype,
                    rngs=nnx.Rngs(init_key),
                )
            )
            self.resume_checkpoint(abs_model)

        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate))

        params = nnx.state(self.model, nnx.Param)
        total_bytes, total_params = memory_usage_params(params)

        if jax.process_index() == 0:
            logger.info(f"Model parameter count: {total_params} using: {total_bytes}")

        self.train_step = nnx.jit(
            functools.partial(rectified_flow_step, training=True),
        )

        self.eval_step = nnx.jit(
            functools.partial(rectified_flow_step, training=False),
        )

        # self.flops_for_step = 0

        if config.using_latents:
            if jax.process_index() == 0:
                logger.info("Loading VAE...")
            self.setup_vae()

    def save_checkpoint(self, global_step: int, eval_loss: Optional[float] = None):

        state = nnx.state(self.model, nnx.Param)

        self.checkpointer.save(
            global_step,
            args=ocp.args.StandardSave(state),
            metrics={
                "eval_loss": eval_loss,
            },
        )
        self.checkpointer.wait_until_finished()

    def resume_checkpoint(self, abs_model):

        abs_state = nnx.state(abs_model, nnx.Param)

        abs_state = jax.tree.map(
            lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
            abs_state,
            nnx.get_named_sharding(abs_state, self.mesh),
        )

        latest_step = self.checkpointer.latest_step()
        if jax.process_index() == 0:
            logger.info(f"Resuming from checkpoint at step {latest_step}...")

        restored_state = self.checkpointer.restore(
            latest_step, args=ocp.args.StandardRestore(abs_state)
        )
        self.checkpointer.wait_until_finished()

        nnx.update(self.model, restored_state)

    def setup_vae(self, vae_path: str = "pcuenq/stable-diffusion-xl-base-1.0-flax"):
        self.vae, self.vae_params = load_pretrained_vae(vae_path, True, subfolder="vae")


def process_batch(
    batch: Any,
    latent_size: int,
    n_channels: int,
    label_field_name: str,
    image_field_name: str,
    using_latents: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process a batch of samples from the dataset.
    Provide the entire batch to the train/eval step, and the in_sharding will partition across
    devices.
    If an image is not square, it will be center cropped to the smaller dimension, before being resized to the latent size.
    """

    images: List[Image.Image] = batch[image_field_name]
    img_mode = "L" if n_channels == 1 else "RGB"
    if not using_latents:
        for i, image in enumerate(images):
            if image.width != image.height:
                smaller_dim = min(image.width, image.height)
                image = center_crop(image, smaller_dim, smaller_dim)
            images[i] = image.resize((latent_size, latent_size)).convert(img_mode)
    image_jnp = jnp.asarray(images, dtype=jnp.float32)
    if using_latents:
        image_jnp = image_jnp.reshape(-1, n_channels, latent_size, latent_size)
    if n_channels == 1:
        image_jnp = image_jnp[:, :, :, None]
    # convert to NCHW format
    if not using_latents:
        image_jnp = image_jnp.transpose((0, 3, 1, 2))
        image_jnp = normalize_images(image_jnp)
    else:
        image_jnp = (image_jnp / 255 - 0.5) * 2
    label = jnp.asarray(batch[label_field_name], dtype=jnp.float32)
    if label.ndim == 2:
        label = label[:, 0]
    return image_jnp, label


def run_eval(
    eval_dataset: Dataset,
    n_eval_batches: int,
    config: Config,
    trainer: Trainer,
    rng: PRNGKey,
    summary_writer: SummaryWriter,
    iter_description_dict: Dict,
    global_step: int,
    do_sample: bool,
    epoch: int,
):
    """
    Run evaluation on the eval subset, and optionally sample the model
    """
    num_eval_batches = 1
    eval_iter = tqdm(
        eval_dataset.iter(batch_size=max(jax.device_count(), 16), drop_last_batch=True),
        leave=False,
        total=num_eval_batches,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )

    for j, eval_batch in enumerate(eval_iter):
        if j >= n_eval_batches:
            break

        # Eval loss
        images, labels = process_batch(
            eval_batch,
            config.latent_size,
            config.n_channels,
            config.label_field_name,
            config.image_field_name,
            config.using_latents,
        )

        images = jax.make_array_from_process_local_data(trainer.data_sharding, images)  # type: ignore
        labels = jax.make_array_from_process_local_data(trainer.data_sharding, labels)  # type: ignore

        images, labels = jax.device_put((images, labels), trainer.data_sharding)

        rng = random.PRNGKey(0)
        rng = jax.device_put(rng, trainer.key_sharding)
        eval_rng, sample_rng = random.split(rng)

        eval_loss = trainer.eval_step(
            trainer.model, trainer.optimizer, images, labels, eval_rng
        )
        iter_description_dict.update({"eval_loss": fmt_float_display(eval_loss)})
        eval_iter.set_postfix(iter_description_dict)
        summary_writer.add_scalar("eval_loss", eval_loss, global_step)

        # Sampling
        if do_sample:
            sample_key, rng = random.split(rng)
            n_labels_to_sample = (
                config.n_labels_to_sample
                if config.n_labels_to_sample
                else config.n_classes
            )
            noise_shape = (
                n_labels_to_sample,
                config.n_channels,
                config.latent_size,
                config.latent_size,
            )
            init_noise = random.normal(sample_rng, noise_shape)
            labels = jnp.arange(0, n_labels_to_sample)
            null_cond = jnp.ones_like(labels) * 10
            samples = rectified_flow_sample(
                trainer.model,
                init_noise,
                labels,
                sample_key,
                decoder=(
                    (trainer.vae, trainer.vae_params) if config.using_latents else None
                ),
                null_cond=null_cond,
                sample_steps=50,
            )
            grid = image_grid(samples)
            sample_img_filename = f"samples/epoch_{epoch}_globalstep_{global_step}.png"
            grid.save(sample_img_filename)

    return eval_loss.item()


def main():
    """
    Arguments:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        eval_save_steps: Number of steps between evaluation runs and checkpoint saves.
        n_eval_batches: Number of batches to evaluate on.
        sample_every_n: Number of epochs between sampling runs.
        dataset_name: Name of the dataset config to select, valid options are in configS.
        profile: Run a single train and eval step, and print out the cost analysis, then exit.
        half_precision: case the model to fp16 for training.
    """

    config = tyro.cli(AllConfigs)
    learning_rate = config.learning_rate
    n_epochs = config.n_epochs
    eval_save_steps = config.eval_save_steps
    n_eval_batches = config.n_eval_batches
    sample_every_n = config.sample_every_n
    profile = config.profile

    dataset: DatasetDict = load_dataset(config.hf_dataset_uri, streaming=False)  # type: ignore
    if not config.eval_split_name:
        config.eval_split_name = "test"
        if config.using_latents:
            dataset = concatenate_datasets([dataset[f"train_{i}"] for i in range(1)])  # type: ignore
            dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        else:
            dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset[config.train_split_name]
    eval_dataset = dataset[config.eval_split_name]

    train_dataset = train_dataset.shard(
        jax.process_count(), jax.process_index()
    )  # shard train dataset across hosts
    eval_dataset = eval_dataset.shard(
        jax.process_count(), jax.process_index()
    )  # shard eval dataset across hosts

    device_count = jax.device_count()
    rng = random.PRNGKey(0)

    trainer = Trainer(
        rng,
        config,
        learning_rate,
        profile,
    )

    summary_writer = SummaryWriter(flush_secs=1, max_queue=1)
    ensure_directory("samples", clear=True)

    iter_description_dict = {"loss": 0.0, "eval_loss": 0.0, "epoch": 0, "step": 0}

    n_samples = len(train_dataset)

    n_evals = 0
    for epoch in range(n_epochs):
        iter_description_dict.update({"epoch": epoch})
        n_batches = n_samples // config.batch_size
        train_iter = tqdm(
            train_dataset.iter(batch_size=config.batch_size, drop_last_batch=True),
            total=n_batches,
            leave=False,
            dynamic_ncols=True,
            disable=jax.process_index() != 0,
        )
        for i, batch in enumerate(train_iter):

            global_step = epoch * n_batches + i

            # Train step
            images, labels = process_batch(
                batch,
                config.latent_size,
                config.n_channels,
                config.label_field_name,
                config.image_field_name,
                config.using_latents,
            )

            images = jax.make_array_from_process_local_data(trainer.data_sharding, images)  # type: ignore
            labels = jax.make_array_from_process_local_data(trainer.data_sharding, labels)  # type: ignore

            images, labels = jax.device_put((images, labels), trainer.data_sharding)

            step_key = random.PRNGKey(global_step)
            step_key = jax.device_put(step_key, trainer.key_sharding)

            if profile:
                # profile_ctx = jax.profiler.trace(
                #     profiler_trace_dir, create_perfetto_link=True
                # )
                profile_ctx = nullcontext()
            else:
                profile_ctx = nullcontext()

            with profile_ctx:
                step_start_time = time.perf_counter()
                train_loss = trainer.train_step(
                    trainer.model, trainer.optimizer, images, labels, step_key
                )

                step_duration = time.perf_counter() - step_start_time
                """
                flops_device_sec = trainer.flops_for_step / (
                    step_duration * device_count
                )

                peak_flops = get_peak_flops()
                mfu = flops_device_sec / peak_flops
                iter_description_dict.update(
                    {
                        "flops_device_sec": fmt_float_display(flops_device_sec),
                        "mfu": fmt_float_display(mfu),
                    }
                )
                """

                summary_writer.add_scalar(
                    "train_step_time",
                    step_duration,
                    global_step,
                )

            iter_description_dict.update(
                {
                    "loss": fmt_float_display(train_loss),
                    "epoch": epoch,
                    "step": i,
                }
            )
            summary_writer.add_scalar("train_loss", train_loss, global_step)

            train_iter.set_postfix(iter_description_dict)

            if global_step % eval_save_steps == 0 or profile:

                eval_loss = run_eval(
                    eval_dataset,
                    n_eval_batches,
                    config,
                    trainer,
                    rng,
                    summary_writer,
                    iter_description_dict,
                    global_step,
                    n_evals % sample_every_n == 0,
                    epoch,
                )

                trainer.save_checkpoint(global_step, eval_loss)

            if profile:
                logger.info("\nExiting after profiling a single step.")
                return

        if epoch % sample_every_n == 0 and not profile:

            run_eval(
                eval_dataset,
                n_eval_batches,
                config,
                trainer,
                rng,
                summary_writer,
                iter_description_dict,
                global_step,
                True,
                epoch,
            )

            trainer.save_checkpoint(global_step, eval_loss)

    trainer.save_checkpoint(global_step)


if __name__ == "__main__":
    main()

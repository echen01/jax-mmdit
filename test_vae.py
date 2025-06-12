import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from datasets import load_dataset
from PIL import Image
from loguru import logger

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")

from vae.vae_flax import load_pretrained_vae

import jax.experimental.compilation_cache.compilation_cache

from streaming.base.format.mds.encodings import Encoding, _encodings
from typing import Any
from streaming import StreamingDataset


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8


vae, params = load_pretrained_vae(
    "pcuenq/stable-diffusion-xl-base-1.0-flax", True, subfolder="vae"
)
sample_size = vae.config.sample_size


@jax.jit
def step(sample: Array):
    return vae.apply(params, sample, method="decode")


remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./local_train_dir"

dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
)


for i, sample in enumerate(dataset):
    encoding = jnp.array(sample["vae_output"])
    sample_tensor = encoding.reshape(1, 4, 32, 32)
    out = step(sample_tensor)
    out_np = np.array(out[0])
    out_np = out_np * 0.5 + 0.5
    img = Image.fromarray(
        (out_np.transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    )
    img.save(f"test_{i}.png")
    if i > 10:
        break

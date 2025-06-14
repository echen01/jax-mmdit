from dataclasses import dataclass, field
from typing import Optional, List
import tyro

from labels import IMAGENET_LABELS_NAMES


@dataclass
class ModelConfig:
    """
    Modifiable params for the model's architecture.
    """

    dim: int
    n_layers: int
    n_heads: int
    patch_size: int = 2


DIT_MODELS = {
    "XL_2": ModelConfig(n_layers=28, dim=1152, patch_size=2, n_heads=16),
    "XL_4": ModelConfig(n_layers=28, dim=1152, patch_size=4, n_heads=16),
    "XL_8": ModelConfig(n_layers=28, dim=1152, patch_size=8, n_heads=16),
    "L_2": ModelConfig(n_layers=24, dim=1024, patch_size=2, n_heads=16),
    "L_4": ModelConfig(n_layers=24, dim=1024, patch_size=4, n_heads=16),
    "L_8": ModelConfig(n_layers=24, dim=1024, patch_size=8, n_heads=16),
    "B_2": ModelConfig(n_layers=12, dim=768, patch_size=2, n_heads=12),
    "B_4": ModelConfig(n_layers=12, dim=768, patch_size=4, n_heads=12),
    "B_8": ModelConfig(n_layers=12, dim=768, patch_size=8, n_heads=12),
    "S_2": ModelConfig(n_layers=12, dim=384, patch_size=2, n_heads=6),
    "S_4": ModelConfig(n_layers=12, dim=384, patch_size=4, n_heads=6),
    "S_8": ModelConfig(n_layers=12, dim=384, patch_size=8, n_heads=6),
}


@dataclass
class Config:
    # Training Options
    n_epochs: int = 100
    learning_rate: float = 1e-4
    eval_save_steps: int = 50
    n_eval_batches: int = 1
    sample_every_n: int = 1
    profile: bool = False
    resume: bool = False
    batch_size: int = 256
    half_precision: bool = False

    # Dataset Options
    hf_dataset_uri: str = "emc348/imagenet-sdxl-vae-uint8"
    n_classes: int = 1000
    latent_size: int = 32
    eval_split_name: Optional[str] = None
    train_split_name: str = "train"
    image_field_name: str = "image"
    label_field_name: str = "label"
    label_names: Optional[List[str]] = None
    n_channels: int = 3
    n_labels_to_sample: Optional[int] = None
    batch_size: int = 256

    # used for streaming datasets
    dataset_length: Optional[int] = None

    model_config: ModelConfig = field(default_factory=lambda: DIT_MODELS["B_2"])

    using_latents: bool = False


DATASET_CONFIGS = {
    "imagenet_vae": Config(
        hf_dataset_uri="emc348/imagenet-sdxl-vae-uint8",
        n_classes=1000,
        latent_size=32,
        n_channels=4,
        label_names=list(IMAGENET_LABELS_NAMES.values()),
        image_field_name="vae_output",
        label_field_name="label",
        n_labels_to_sample=10,
        eval_split_name=None,
        batch_size=128,
        model_config=DIT_MODELS["XL_2"],
        using_latents=True,
    ),
    # https://huggingface.co/datasets/zh-plus/tiny-imagenet
    "imagenet": Config(
        hf_dataset_uri="evanarlian/imagenet_1k_resized_256",
        n_classes=1000,
        latent_size=32,
        n_channels=3,
        dataset_length=1281167,
        label_names=list(IMAGENET_LABELS_NAMES.values()),
        image_field_name="image",
        label_field_name="label",
        n_labels_to_sample=10,
        eval_split_name="val",
        batch_size=154 * 4,
        model_config=DIT_MODELS["XL_2"],
    ),
    # https://huggingface.co/datasets/cifar10
    "cifar10": Config(
        hf_dataset_uri="cifar10",
        n_classes=10,
        image_field_name="img",
        latent_size=32,
        label_names=[
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        eval_split_name="test",
        dataset_length=50000,
        model_config=DIT_MODELS["B_2"],
        batch_size=96,
    ),
    # TODO find the class counts and resize with preprocessor
    "butterflies": Config(
        hf_dataset_uri="ceyda/smithsonian_butterflies",
        n_channels=3,
        n_classes=25,
        latent_size=64,
    ),
    "mnist": Config(
        hf_dataset_uri="ylecun/mnist",
        n_channels=1,
        n_classes=10,
        latent_size=28,
        batch_size=256,  # 796 * 4,
        eval_split_name="test",
        dataset_length=60000,
    ),
    "flowers": Config(
        hf_dataset_uri="nelorth/oxford-flowers",
        n_channels=3,
        n_classes=102,
        latent_size=32,
        batch_size=256,
        n_labels_to_sample=16,
        eval_split_name="test",
        model_config=ModelConfig(dim=64, n_layers=10, n_heads=8),
    ),
    "fashion_mnist": Config(
        hf_dataset_uri="fashion_mnist",
        n_channels=1,
        n_classes=10,
        latent_size=28,
        label_names=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    ),
}

AllConfigs = tyro.extras.subcommand_type_from_defaults(DATASET_CONFIGS)

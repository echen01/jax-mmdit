import numpy as np
import jax
from loguru import logger

jax.experimental.compilation_cache.compilation_cache.set_cache_dir("jit_cache")

import jax.experimental.compilation_cache.compilation_cache

from streaming.base.format.mds.encodings import Encoding, _encodings
from typing import Any
from streaming import StreamingDataset
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


"""
Converts the Imagenet int8 dataset from Simo Ryu to Parquet format with uint8 encoding.
Can be used as a standard Hugging Face dataset without the MosaicML StreamingDataset format.
"""


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8)
        return x


_encodings["uint8"] = uint8


remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./local_train_dir"

dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
    split=None,
    shuffle=False,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
)

train_dataset: Dataset = Dataset.from_dict({"label": [], "vae_output": []})
train_dataset.set_format(type="numpy")
new_rows = []
count = 0
for i, sample in enumerate(dataset):

    new_sample = {
        "label": int(sample["label"]),
        "vae_output": sample["vae_output"],
    }
    new_rows.append(new_sample)

    # 524300
    if i % 500000 == 0 and i > 0:
        logger.info(f"Uploading at iteration {i}...")
        dataset_new_rows = Dataset.from_list(new_rows)
        # concat_dataset = concatenate_datasets([train_dataset, dataset_new_rows])

        dataset = DatasetDict({f"train_{count}": dataset_new_rows})
        dataset.push_to_hub("emc348/imagenet-sdxl-vae-uint8")
        new_rows = []

        count += 1
dataset_new_rows = Dataset.from_list(new_rows)
concat_dataset = concatenate_datasets([train_dataset, dataset_new_rows])

dataset = DatasetDict({"train": concat_dataset})
dataset.push_to_hub("emc348/imagenet-sdxl-vae-uint8")

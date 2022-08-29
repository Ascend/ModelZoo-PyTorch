# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from shutil import copyfile

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

KEYS_MAPPING = {
    "checkpoint_callback_best_model_score": (ModelCheckpoint, "best_model_score"),
    "checkpoint_callback_best_model_path": (ModelCheckpoint, "best_model_path"),
    "checkpoint_callback_best": (ModelCheckpoint, "best_model_score"),
    "early_stop_callback_wait": (EarlyStopping, "wait_count"),
    "early_stop_callback_patience": (EarlyStopping, "patience"),
}


def upgrade_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    checkpoint["callbacks"] = checkpoint.get("callbacks") or {}

    for key, new_path in KEYS_MAPPING.items():
        if key in checkpoint:
            value = checkpoint[key]
            callback_type, callback_key = new_path
            checkpoint["callbacks"][callback_type] = checkpoint["callbacks"].get(callback_type) or {}
            checkpoint["callbacks"][callback_type][callback_key] = value
            del checkpoint[key]

    torch.save(checkpoint, filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Upgrade an old checkpoint to the current schema. \
        This will also save a backup of the original file."
    )
    parser.add_argument("--file", help="filepath for a checkpoint to upgrade")

    args = parser.parse_args()

    log.info("Creating a backup of the existing checkpoint file before overwriting in the upgrade process.")
    copyfile(args.file, args.file + ".bak")
    upgrade_checkpoint(args.file)

# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import logging
import os
from pathlib import Path

from coco_format_utils import COCO_json
from data_format_transforms import transform_bbox_s2s_to_coco
from data_utils import S2S_ORIGINAL_CATEGORIES as ORIGINAL_CATEGORIES
from data_utils import (
    create_category_txt_filepaths,
    extract_json_data,
    load_all_images_paths_from_txt,
    merge_train_test_subsets,
    save_json,
)

### HARDCODED
train_filename = "train_data.txt"
test_filename = "test_data.txt"
train_all_filename = "train_all.txt"
coco_json_save_name = "all_street_train.json"


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to transform original Exact Street2Shop annotations to COCO format."
    )
    parser.add_argument(
        "--root-dir-path",
        help="path to root directory of Steet2Shop data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata-dir",
        help="directory name with Steet2Shop metadata",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--images-dir",
        help="directory name with Steet2Shop images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="directory name where all new data will be saved",
        type=str,
        required=True,
    )

    ### PARAMETERS
    args = parser.parse_args()

    root_dir = Path(args.root_dir_path)
    meta_dir = root_dir / args.metadata_dir
    images_dir = root_dir / args.images_dir
    save_dir = root_dir / args.save_dir
    train_file_path = save_dir / train_all_filename
    coco_json_save_path = save_dir / coco_json_save_name

    save_dir.mkdir(exist_ok=True, parents=False)

    categories_dict = {
        cat_name: num for num, cat_name in enumerate(ORIGINAL_CATEGORIES)
    }

    # Create single txt files with names of all images in train / test set
    extract_json_data(
        jsons_path=meta_dir / "json",
        save_dir=save_dir,
        save_filename=train_filename,
        key_name="photo",
        ext="jpg",
        mode="train",
    )
    extract_json_data(
        jsons_path=meta_dir / "json",
        save_dir=save_dir,
        save_filename=test_filename,
        key_name="photo",
        ext="jpg",
        mode="test",
    )

    # Merge train and test into one txt file
    merge_train_test_subsets(
        filenames=[train_filename, test_filename],
        save_dir=save_dir,
        save_filename=train_all_filename,
    )
    images_names = load_all_images_paths_from_txt(path=train_file_path)
    create_category_txt_filepaths(
        categories_dict=categories_dict,
        meta_dir=meta_dir,
        save_dir=save_dir,
        mode="single",
    )

    # Create COCO format annotations
    coco_json = COCO_json(
        images_dir=images_dir,
        save_dir=save_dir,
        categories_dict=categories_dict,
        sets=["train", "test"],
        images_names=images_names,
        meta_dir=meta_dir,
    )
    coco_json.create_full_coco_json(bbox_transform_func=transform_bbox_s2s_to_coco)
    save_json(coco_json.json, coco_json_save_path, mode="w")
    log.info("Street2Shop_to_coco processing finished")

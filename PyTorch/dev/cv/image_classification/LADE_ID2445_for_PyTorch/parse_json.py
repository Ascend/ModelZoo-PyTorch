#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
import json, os
import argparse
from tqdm import tqdm
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--file",
        help="json file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args

def convert(json_file, image_root):
    all_annos = json.load(open(json_file, 'r'))
    import ipdb; ipdb.set_trace()
    annos = all_annos['annotations']
    images = all_annos['images']
    new_annos = []

    print("Converting file {} ...".format(json_file))
    for anno, image in tqdm(zip(annos, images)):
        assert image["id"] == anno["id"]

        new_annos.append({"image_id": image["id"],
                          "im_height": image["height"],
                          "im_width": image["width"],
                          "category_id": anno["category_id"],
                          "fpath": os.path.join(image_root, image["file_name"])})
    num_classes = len(all_annos["categories"])
    return {"annotations": new_annos,
            "num_classes": num_classes}

if __name__ == "__main__":
    args = parse_args()
    converted_annos = convert(args.file, args.root)
    save_path = os.path.join(args.sp, "converted_" + os.path.split(args.file)[-1])
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)
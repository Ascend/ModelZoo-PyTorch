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
import concurrent
from PIL import Image, ImageFile
import concurrent.futures

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

success = 0
failed = 0


def transform_image(source):
    try:
        im = Image.open(source)
        rgb_im = im.convert("RGB")
        new_name = target_dir / Path(source).with_suffix(".jpg").name
        rgb_im.save(new_name)
        global success
        success += 1
    except Exception as e:
        log.error(e)
        global failed
        failed += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to transform all images in a folder to jpg."
    )
    parser.add_argument(
        "--source-dir-path",
        help="path to directory with source images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target-dir-path",
        help="path to directory where jpg images will be saved. Must be different from source-dir-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-threads",
        help="number of threads used for converting images",
        type=int,
        required=False,
        default=4,
    )

    args = parser.parse_args()
    source_dir = Path(args.source_dir_path)
    target_dir = Path(args.target_dir_path)
    log.info(f"Creating target dir: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    images_source_list = [source_dir / item for item in os.listdir(source_dir)]
    log.info(
        f"Processing images dir: {source_dir}\nNumber of images to process: {len(images_source_list)}"
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_threads
    ) as executor:
        executor.map(transform_image, images_source_list)

    log.info("Processing finished.")
    log.info(f"Sucessful images: {success}")
    log.info(f"Failed images: {failed}")

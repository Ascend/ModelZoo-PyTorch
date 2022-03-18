# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Prepare SBU Shadow datasets"""
import os
import sys
import argparse
import zipfile

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from segmentron.utils import download, makedirs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize SBU Shadow dataset.',
        epilog='Example: python sbu_shadow.py --download-dir ~/SBU-shadow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args


#####################################################################################
# Download and extract SBU shadow datasets into ``path``

def download_sbu(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip'),
    ]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite)
        # extract
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall(path=path)
        print("Extracted", filename)


if __name__ == '__main__':
    args = parse_args()
    default_dir = os.path.join(root_path, 'datasets/sbu')
    if args.download_dir is not None:
        _TARGET_DIR = args.download_dir
    else:
        _TARGET_DIR = default_dir
    makedirs(_TARGET_DIR)
    if os.path.exists(default_dir):
        print('{} is already exist!'.format(default_dir))
    else:
        try:
            os.symlink(_TARGET_DIR, default_dir)
        except Exception as e:
            print(e)
        download_sbu(_TARGET_DIR, overwrite=False)

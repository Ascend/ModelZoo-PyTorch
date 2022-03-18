# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division

import wget
import os
import shutil
import zipfile
import sys


def download(url, filename):
    r"""Download file from the internet.
    
    Args:
        url (string): URL of the internet file.
        filename (string): Path to store the downloaded file.
    """
    return wget.download(url, out=filename)


def extract(filename, extract_dir):
    r"""Extract zip file.
    
    Args:
        filename (string): Path of the zip file.
        extract_dir (string): Directory to store the extracted results.
    """
    if os.path.splitext(filename)[1] == '.zip':
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.'.format(
            os.path.splitext(filename)[1]), filename)


def compress(dirname, save_file):
    """Compress a folder to a zip file.
    
    Arguments:
        dirname {string} -- Directory of all files to be compressed.
        save_file {string} -- Path to store the zip file.
    """
    shutil.make_archive(save_file, 'zip', dirname)

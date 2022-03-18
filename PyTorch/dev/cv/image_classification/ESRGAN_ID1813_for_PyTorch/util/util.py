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
import os
import shutil
import urllib
import zipfile
import tarfile

from tqdm import tqdm
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


class LambdaLR:
    def __init__(self, n_epoch, offset, total_batch_size, decay_batch_size):
        self.n_epoch = n_epoch
        self.offset = offset
        self.total_batch_size = total_batch_size
        self.decay_batch_size = decay_batch_size

    def step(self, epoch):
        factor = pow(0.5, int(((self.offset + epoch) * self.total_batch_size) / self.decay_batch_size))
        return factor


def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(zip_path)


def unzip_tar_file(zip_path, data_path):
    tar_ref = tarfile.open(zip_path, "r:")
    tar_ref.extractall(data_path)
    tar_ref.close()
    os.remove(zip_path)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def reformat_file(dataset_path):
    dir = os.listdir(dataset_path)

    for data_scale_dir in dir:
        data_scale_dir = os.path.join(dataset_path, data_scale_dir)
        data_file = os.listdir(data_scale_dir)

        print(f"[!] Move file in {data_scale_dir}")
        for file in tqdm(data_file):
            src_path = os.path.join(data_scale_dir, file)
            des_path = os.path.join(dataset_path, file)
            shutil.move(src_path, des_path)
        os.rmdir(data_scale_dir)

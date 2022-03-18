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
from facenet_pytorch import MTCNN, training
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import time


def main():
    device = 'npu' if torch.npu.is_available() else 'cpu'
    print(f'Running on device "{device}"')

    mtcnn = MTCNN(device=device)

    batch_size = 32

    # Generate data loader
    ds = datasets.ImageFolder(
        root='data/test_images/',
        transform=transforms.Resize((512, 512))
    )
    dl = DataLoader(
        dataset=ds,
        num_workers=4,
        collate_fn=training.collate_pil,
        batch_size=batch_size,
        sampler=RandomSampler(ds, replacement=True, num_samples=960),
    )

    start = time.time()
    faces = []
    for x, _ in tqdm(dl):
        faces.extend(mtcnn(x))
    elapsed = time.time() - start
    print(f'Elapsed: {elapsed} | EPS: {len(dl) * batch_size / elapsed}')


if __name__ == '__main__':
    main()

# coding=utf-8
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# ==========================================================================

import os
import torch
import torch_npu
import PIL

from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torch_npu.contrib import transfer_to_npu

torch.npu.set_compile_mode(jit=False)

device = "npu"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    image_grid_size = Image.new("RGB",size=(cols * w, rows * h))
    image_grid_size_w, image_grid_size_h = image_grid_size.size


    for i, img in enumerate(imgs):
        image_grid_size.paste(img,box=(i % cols * w, i // cols * h))
    return image_grid_size

pretrained_model_name = "/stable-diffusion-v1-5"
repo_id = "/cat-toy"

prompt = "a grafitti in a favela wall with a <cat-toy> on it"

pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name).to(device)

pipeline.load_textual_inversion(repo_id)

num_samples = 2
num_rows = 2

all_images = []

for _ in range(num_rows):
    images = pipeline(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, gudiance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images,num_samples,num_rows)
grid.save("./grad.png")
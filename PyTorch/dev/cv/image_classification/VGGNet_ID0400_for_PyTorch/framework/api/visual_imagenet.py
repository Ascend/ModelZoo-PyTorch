# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
# ==============================================================================
import json
import os
import urllib.request

import torch
import torchvision.transforms as transforms
from PIL import Image
from django.shortcuts import render
from rest_framework.views import APIView

from vgg_pytorch import VGG
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

model = VGG.from_pretrained("vgg19")
# move the model to GPU for speed if available
model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
# switch to evaluate mode
model.eval()


def preprocess(filename, label):
    input_image = Image.open(filename)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    labels_map = json.load(open(label))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    return input_batch, labels_map


def index(request):
    r""" Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
      request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.

    Return:
      Base64 bit encoding of the image.

    Notes:
      Later versions will not contexturn an image's address,
      but instead a base64-bit encoded address
    """

    return render(request, "index.html")


class IMAGENET(APIView):

    @staticmethod
    def get(request):
        """ Get the image based on the base64 encoding or url address

        Args:
          request: Post request in url.
            - image_code: 64-bit encoding of images.
            - url:        The URL of the image.

        Return:
          Base64 bit encoding of the image.

        Notes:
          Later versions will not contexturn an image's address,
          but instead a base64-bit encoded address
        """

        base_path = "static/imagenet"

        try:
            os.makedirs(base_path)
        except OSError:
            pass

        filename = os.path.join(base_path, "imagenet.png")
        if os.path.exists(filename):
            os.remove(filename)

        context = {
            "status_code": 20000
        }
        return render(request, "imagenet.html", context)

    @staticmethod
    def post(request):
        """ Get the image based on the base64 encoding or url address
        Args:
            request: Post request in url.
            - image_code: 64-bit encoding of images.
            - url:        The URL of the image.

        Return:
            Base64 bit encoding of the image.

        Notes:
            Later versions will not contexturn an image's address,
            but instead a base64-bit encoded address
        """

        context = None

        # Get the url for the image
        url = request.POST.get("url")
        base_path = "static/imagenet"
        data_path = "data"

        try:
            os.makedirs(base_path)
        except OSError:
            pass

        filename = os.path.join(base_path, "imagenet.png")
        label = os.path.join(data_path, "labels_map.txt")

        image = urllib.request.urlopen(url)
        with open(filename, "wb") as v:
            v.write(image.read())

        image, labels_map = preprocess(filename, label)
        image = image.to(f'npu:{NPU_CALCULATE_DEVICE}')

        with torch.no_grad():
            logits = model(image)
        preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

        for idx in preds:
            label = labels_map[idx]
            probability = torch.softmax(logits, dim=1)[0, idx].item() * 100
            probability = str(probability)[:5]

            context = {
                "status_code": 20000,
                "message": "OK",
                "filename": filename,
                "label": label,
                "probability": probability}
        return render(request, "imagenet.html", context)

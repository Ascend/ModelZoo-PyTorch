# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import argparse
import numpy as np
import onnxruntime
from PIL import Image
import torch
from clip.utils import _MODEL_INFO, image_transform

def clip_infer(model_arch, model_path, src_path, npy_path):
    img_session = onnxruntime.InferenceSession(model_path)
    model_arch = "ViT-H-14" 
    preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
    for image_path in os.listdir(src_path):
        image = preprocess(Image.open(os.path.join(src_path, image_path))).unsqueeze(0)
        image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] 
        image_features = torch.tensor(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        np.save(os.path.join(npy_path, os.path.splitext(os.path.basename(image_path))[0]), image_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, default='ViT-H-14', help='model skelecton.')
    parser.add_argument('--model_path', type=str, default='./models/vit-h-14.img.fp16.onnx', help='path to model.')
    parser.add_argument('--src_path', type=str, default='./Chinese-CLIP/examples', help='path to images.')
    parser.add_argument('--npy_path', type=str, default='./onnx_npy_path', help='path to save npy files.')
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        os.makedirs(args.npy_path)

    clip_infer(args.model_arch, args.model_path, args.src_path, args.npy_path)

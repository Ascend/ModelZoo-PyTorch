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

import argparse

from PIL import Image
import torch
from open_clip import create_model_and_transforms, get_tokenizer


def torch2onnx(model_name, pretrained, image, text, opset_version, onnx_prefix):
    model, _, preprocess = create_model_and_transforms(model_name,
                                                       pretrained=pretrained)
    tokenizer = get_tokenizer(model_name)
    vision_input = preprocess(Image.open(image)).unsqueeze(0)
    text_input = tokenizer([text])

    # convert vision ONNX model
    vision_onnx_path = f'{onnx_prefix}_vision.onnx'
    torch.onnx.export(model,
                      (vision_input, None),
                      vision_onnx_path,
                      input_names=['image'],
                      output_names=['image_embed'],
                      export_params=True,
                      opset_version=opset_version,
                      verbose=False,
                      dynamic_axes={'image': {0: '-1'}, 'image_embed': {0: '-1'}})

    # convert text ONNX model
    text_onnx_path = f'{onnx_prefix}_text.onnx'
    torch.onnx.export(model,
                      (None, text_input),
                      text_onnx_path,
                      input_names=['text'],
                      output_names=['text_embed'],
                      export_params=True,
                      opset_version=opset_version,
                      verbose=False,
                      dynamic_axes={'text': {0: '-1'}, 'text_embed': {0: '-1'}})


def main():
    parser = argparse.ArgumentParser(description='Convert OpenCLIP model to ONNX.')
    parser.add_argument('--model-name', type=str,
                        default='ViT-B-32',
                        help='Specify the model name to be converted.')
    parser.add_argument('--pretrained', type=str,
                        default='laion2b_s34b_b79k',
                        help='Specify the pretrained weights.')
    parser.add_argument('--image', type=str,
                        default='./open_clip/docs/CLIP.png',
                        help='Path to one image for jit tracing.')
    parser.add_argument('--text', type=str,
                        default='a dog',
                        help='Specify one text for jit tracing.')
    parser.add_argument('--opset-version', type=int,
                        default=11,
                        help='ONNX opset version.')
    parser.add_argument('--onnx-prefix', type=str,
                        default='models/vit_b_32',
                        help='Path(prefix) of the output ONNX.')
    args = parser.parse_args()

    torch2onnx(args.model_name, args.pretrained, args.image, args.text, 
               args.opset_version, args.onnx_prefix)


if __name__ == '__main__':
    main()

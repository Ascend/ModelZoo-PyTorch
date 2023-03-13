"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import torch
import timm
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Path', add_help=False)
    parser.add_argument('--model_path', required=True, metavar='DIR',
                        help='path to model')
    parser.add_argument('--save_dir', default="models/onnx", type=str,
                        help='save dir for onnx model')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--model_name', required=True, type=str,
                        help='model name for ViT')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    model = timm.create_model(args.model_name)
    model.load_pretrained(args.model_path)
    model.eval()
    input_size = int(args.model_name[-3:])
    tensor = torch.zeros(args.batch_size, 3, input_size, input_size)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir,
                             f"{args.model_name}_bs{args.batch_size}.onnx")
    torch.onnx.export(model, tensor, save_path, opset_version=11,
                      do_constant_folding=True, input_names=["input"], output_names=["output"])


if __name__ == "__main__":
    main()

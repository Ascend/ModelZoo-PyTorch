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
import sys
import ast
import cv2
import argparse
import onnxruntime
import numpy as np
from segment_anything.utils.transforms import ResizeLongestSide
from sam_preprocessing_pytorch import encoder_preprocessing


preprocessing_config = {
    'image_size': 1024,
    'mean':[123.675, 116.28, 103.53],
    'std':[58.395, 57.12, 57.375],
}


def sam_encoder_quant_proprecessing(src_path, encoder_quant_save_path): 
    image = cv2.imread(src_path)   
    x = encoder_preprocessing(image)
    if not os.path.exists(encoder_quant_save_path + "/x"):
        os.makedirs(os.path.realpath(encoder_quant_save_path + "/x"), mode=0o744)
    x.tofile(encoder_quant_save_path + "/x/" + "x.bin")


def sam_decoder_quant_proprecessing(src_path, encoder_onnx_model_path, decoder_quant_save_path, input_point):
    image = cv2.imread(src_path)
    encoder_session = onnxruntime.InferenceSession(encoder_onnx_model_path)
    x = encoder_preprocessing(image)
    encoder_inputs = {
        'x':x,
    }
    output = encoder_session.run(None, encoder_inputs)
    if not os.path.exists(decoder_quant_save_path + "/image_embedding"):
        os.makedirs(os.path.realpath(decoder_quant_save_path + "/image_embedding"), mode=0o744)
    output[0].tofile(decoder_quant_save_path + "/image_embedding/" + "image_embedding.bin")
    input_point = np.array(input_point)
    input_label = [1] * len(input_point)
    input_label = np.array(input_label)
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    transform = ResizeLongestSide(preprocessing_config['image_size'])
    onnx_coord = transform.apply_coords(onnx_coord, image.shape[: 2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    if not os.path.exists(decoder_quant_save_path + "/point_coord"):
        os.makedirs(os.path.realpath(decoder_quant_save_path + "/point_coord"), mode=0o744)
    if not os.path.exists(decoder_quant_save_path + "/point_label"):
        os.makedirs(os.path.realpath(decoder_quant_save_path + "/point_label"), mode=0o744)
    if not os.path.exists(decoder_quant_save_path + "/mask_input"):
        os.makedirs(os.path.realpath(decoder_quant_save_path + "/mask_input"), mode=0o744)
    if not os.path.exists(decoder_quant_save_path + "/has_mask_input"):
        os.makedirs(os.path.realpath(decoder_quant_save_path + "/has_mask_input"), mode=0o744)
    onnx_coord.tofile(decoder_quant_save_path + "/point_coord/" + "point_coord.bin")
    onnx_label.tofile(decoder_quant_save_path + "/point_label/" + "point_label.bin")
    onnx_mask_input.tofile(decoder_quant_save_path + "/mask_input/" + "mask_input.bin")
    onnx_has_mask_input.tofile(decoder_quant_save_path + "/has_mask_input/" + "has_mask_input.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, default='./data/demo.jpg', help='input path to image')
    parser.add_argument('--encoder-quant-save-path', type=str, default='./encoder_quant_bin', help='data path to quant')
    parser.add_argument('--encoder-onnx-model-path', type=str, default='./models/encoder.onnx', help='path to encoder onnx model')
    parser.add_argument('--decoder-quant-save-path', type=str, default='./decoder_quant_bin', help='data path to quant')
    parser.add_argument('--input-point', type=ast.literal_eval, default=[[500, 375], [1125, 625], [1520, 625]], help='input points.')
    parser.add_argument('--encoder-quant', action='store_true', help='if True, will generate quantization data.')
    parser.add_argument('--decoder-quant', action='store_true', help='if True, will generate quantization data.')
    args = parser.parse_args()

    if not os.path.exists(args.encoder_quant_save_path):
        os.makedirs(os.path.realpath(args.encoder_quant_save_path), mode=0o744)

    if not os.path.exists(args.decoder_quant_save_path):
        os.makedirs(os.path.realpath(args.decoder_quant_save_path), mode=0o744)

    if args.encoder_quant:
        sam_encoder_quant_proprecessing(args.src_path, args.encoder_quant_save_path)
    elif args.decoder_quant:
        sam_decoder_quant_proprecessing(args.src_path, args.encoder_onnx_model_path, args.decoder_quant_save_path, args.input_point)


if __name__ == '__main__':
    main()
   
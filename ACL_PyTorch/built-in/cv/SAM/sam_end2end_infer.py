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
import numpy as np
from ais_bench.infer.interface import InferSession
from sam_preprocessing_pytorch import encoder_preprocessing, decoder_preprocessing
from sam_postprocessing_pytorch import sam_postprocessing


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [ int(v) for v in value.split(',') ]
        for ivalue in ilist[:2]:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("{} of device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, value, min_value, max_value))
        return ilist[:2]
    else:
		# default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
                ivalue, min_value, max_value))
        return ivalue
    

def save_mask(mask, image, src_path, save_path, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0.1176, 0.5647, 1, 0.6])
    h, w = mask.shape[-2: ]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask_image[:, :, 0:3]
    mask_img = mask_image * 200 + image
    image_name = src_path.split('/')[-1].split('.')[0]
    cv2.imwrite(save_path + "/" + image_name + "_result.jpg", mask_img)


def encoder_infer(session_encoder, x):
    encoder_outputs = session_encoder.infer([x])
    image_embedding = encoder_outputs[0]
    return image_embedding


def decoder_infer(session_decoder, decoder_inputs):
    decoder_outputs = session_decoder.infer(decoder_inputs, mode="dymdims", custom_sizes=[1000, 1000000])
    low_res_masks = decoder_outputs[1]
    return low_res_masks


def sam_infer(src_path, session_encoder, session_decoder, input_point, save_path):
    image = cv2.imread(src_path)
    x = encoder_preprocessing(image)
    image_embedding = encoder_infer(session_encoder, x)
    decoder_inputs = decoder_preprocessing(image_embedding, input_point, image)
    low_res_masks = decoder_infer(session_decoder, decoder_inputs)
    masks = sam_postprocessing(low_res_masks, image)
    save_mask(masks, image, src_path, save_path, random_color=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, default='./datasets/demo.jpg', help='input path to image')
    parser.add_argument('--save-path', type=str, default='./outputs', help='output path to image')
    parser.add_argument('--encoder-model-path', type=str, default='./models/encoder.om', help='path to encoder model')
    parser.add_argument('--decoder-model-path', type=str, default='./models/decoder.om', help='path to decoder model')
    parser.add_argument('--input-point', type=ast.literal_eval, default=[[500, 375], [1125, 625], [1520, 625]], help='input points')
    parser.add_argument('--device-id', type=check_device_range_valid, default=0, help='NPU device id. Give 2 ids to enable parallel inferencing.')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(os.path.realpath(args.save_path), mode=0o744)

    session_encoder = InferSession(args.device_id, args.encoder_model_path)
    session_decoder = InferSession(args.device_id, args.decoder_model_path)

    sam_infer(args.src_path, session_encoder, session_decoder, args.input_point, args.save_path)


if __name__ == '__main__':
    main()
   
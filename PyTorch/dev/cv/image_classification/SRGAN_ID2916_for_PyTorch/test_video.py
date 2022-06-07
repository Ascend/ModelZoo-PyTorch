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
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from model import Generator
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--video_name', type=str, help='test low resolution video name')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.npu.is_available():
        model = model.npu()
    # for cpu
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    videoCapture = cv2.VideoCapture(VIDEO_NAME)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    sr_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR),
                     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR)
    compared_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10),
                           int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * UPSCALE_FACTOR + 10 + int(
                               int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR * 2 + 10) / int(
                                   10 * int(int(
                                       videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR) // 5 + 1)) * int(
                                   int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * UPSCALE_FACTOR) // 5 - 9)))
    output_sr_name = 'out_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    output_compared_name = 'compare_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    sr_video_writer = cv2.VideoWriter(output_sr_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, sr_video_size)
    compared_video_writer = cv2.VideoWriter(output_compared_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps,
                                            compared_video_size)
    # read frame
    success, frame = videoCapture.read()
    test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')
    for index in test_bar:
        if success:
            image = Variable(ToTensor()(frame), volatile=True).unsqueeze(0)
            if torch.npu.is_available():
                image = image.npu()

            out = model(image)
            out = out.cpu()
            out_img = out.data[0].numpy()
            out_img *= 255.0
            out_img = (np.uint8(out_img)).transpose((1, 2, 0))
            # save sr video
            sr_video_writer.write(out_img)

            # make compared video and crop shot of left top\right top\center\left bottom\right bottom
            out_img = ToPILImage()(out_img)
            crop_out_imgs = transforms.FiveCrop(size=out_img.width // 5 - 9)(out_img)
            crop_out_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_out_imgs]
            out_img = transforms.Pad(padding=(5, 0, 0, 5))(out_img)
            compared_img = transforms.Resize(size=(sr_video_size[1], sr_video_size[0]), interpolation=Image.BICUBIC)(
                ToPILImage()(frame))
            crop_compared_imgs = transforms.FiveCrop(size=compared_img.width // 5 - 9)(compared_img)
            crop_compared_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compared_imgs]
            compared_img = transforms.Pad(padding=(0, 0, 5, 5))(compared_img)
            # concatenate all the pictures to one single picture
            top_image = np.concatenate((np.asarray(compared_img), np.asarray(out_img)), axis=1)
            bottom_image = np.concatenate(crop_compared_imgs + crop_out_imgs, axis=1)
            bottom_image = np.asarray(transforms.Resize(
                size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(
                ToPILImage()(bottom_image)))
            final_image = np.concatenate((top_image, bottom_image))
            # save compared video
            compared_video_writer.write(final_image)
            # next frame
            success, frame = videoCapture.read()

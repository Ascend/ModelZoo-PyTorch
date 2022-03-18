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
# Import Dependencies
import os
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model import ESPCN
from utils import calculate_psnr
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def prepare_image(hr_image, device):
    """ Function to prepare hr/lr/bicubic images for inference and performance comparison

        hr_image: high resolution input image
        device: 'cpu', 'cuda'
        returns: lr/hr Y channel, bicubic YCbCr and original Bicubic Images
    """

    # Load HR image: rH x rW x C, r: scaling factor
    hr_width = (hr_image.width // args.scaling_factor) * args.scaling_factor
    hr_height = (hr_image.height // args.scaling_factor) * args.scaling_factor
    hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)

    # LR Image: H x W x C
    # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
    lr_image = hr_image.resize((hr_image.width // args.scaling_factor, hr_image.height // args.scaling_factor),
                               resample=Image.BICUBIC)

    # Generate Bicubic image for performance comparison
    bicubic_image = lr_image.resize((lr_image.width * args.scaling_factor, lr_image.height * args.scaling_factor),
                                    resample=Image.BICUBIC)
    bicubic_image.save(os.path.join(
        args.dirpath_out,
        os.path.basename(args.fpath_image).replace(".png", f"_bicubic_x{args.scaling_factor}.png")
    ))

    # Convert PIL image to numpy array
    hr_image = np.array(hr_image).astype(np.float32)
    lr_image = np.array(lr_image).astype(np.float32)
    bicubic_image = np.array(bicubic_image).astype(np.float32)

    # Convert RGB to YCbCr
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
    bicubic_image_ycrcb = cv2.cvtColor(bicubic_image, cv2.COLOR_RGB2YCrCb)

    # As per paper, using only the luminescence channel gave the best outcome
    hr_y = hr_image[:, :, 0]
    lr_y = lr_image[:, :, 0]

    # Normalize images
    hr_y /= 255.
    lr_y /= 255.
    bicubic_image /= 255.

    # Convert Numpy to Torch Tensor and send to device
    hr_y = torch.from_numpy(hr_y).to(f'npu:{NPU_CALCULATE_DEVICE}')
    hr_y = hr_y.unsqueeze(0).unsqueeze(0)

    lr_y = torch.from_numpy(lr_y).to(f'npu:{NPU_CALCULATE_DEVICE}')
    lr_y = lr_y.unsqueeze(0).unsqueeze(0)

    return lr_y, hr_y, bicubic_image_ycrcb, bicubic_image


def infer(args):
    """ Function to perform inference on test images

        args
    """

    # Select Device
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    cudnn.benchmark = True

    # Load Model
    model = ESPCN(num_channels=1, scaling_factor=args.scaling_factor)
    model.load_state_dict(torch.load(args.fpath_weights))
    model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    model.eval()

    # Read & Prepare Image for Inference
    # Load HR image: rH x rW x C, r: scaling factor
    hr_image = Image.open(args.fpath_image).convert('RGB')
    lr_y, hr_y, ycbcr, bicubic_image = prepare_image(hr_image, device)

    with torch.no_grad():
        preds = model(lr_y)

    psnr_hr_sr = calculate_psnr(hr_y, preds)
    print('PSNR (HR/SR): {:.2f}'.format(psnr_hr_sr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB), 0.0, 255.0).astype(np.uint8)
    output = Image.fromarray(output)
    output.save(os.path.join(
        args.dirpath_out,
        os.path.basename(args.fpath_image).replace(".png", f"_espcn_x{args.scaling_factor}.png")
    ))

    # Plot Image Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.array(hr_image))
    ax1.set_title("HR Image")
    ax2.imshow(bicubic_image)
    ax2.set_title("Bicubic Image x3")
    ax3.imshow(np.array(output))
    ax3.set_title("SR Image x3 (PSNR: {:.2f} dB)".format(psnr_hr_sr))
    fig.suptitle('ESPCN Single Image Super Resolution')
    plt.show()
    fig.set_size_inches(20, 10, forward=True)
    fig.savefig(os.path.join(args.dirpath_out, "result.png"), dpi=100)


def build_parser():
    parser = ArgumentParser(prog="ESPCN Inference")
    parser.add_argument("-w", "--fpath_weights", required=True, type=str,
                        help="Required. Path to trained model weights.")
    parser.add_argument("-i", "--fpath_image", required=True, type=str,
                        help="Required. Path to image file to perform inference on.")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Required. Path to output image directory.")
    parser.add_argument("-sf", "--scaling_factor", default=3, required=False, type=int,
                        help="Optional. Image Up-scaling factor.")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    infer(args)

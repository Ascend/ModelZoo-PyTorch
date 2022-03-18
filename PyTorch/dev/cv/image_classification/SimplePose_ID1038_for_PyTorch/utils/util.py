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
import numpy as np
import math
import numbers
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from io import StringIO
import PIL.Image


def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4))  # B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4  # G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  # B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375))  # R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  # G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5)  # R: 1 ~ 0.5
    return c


def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x, :] = getJetColor(gray_img[y, x], 0, 1)
    return out


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    # 娉ㄦ剰! concatenate 涓や釜鏁扮粍鐨勯『搴忓緢閲嶈
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def center_pad(image, stride, padValue):
    w, h = image.shape[1], image.shape[0]
    delta_h = 0 if (h % stride == 0) else stride - (h % stride)
    delta_w = 0 if (w % stride == 0) else stride - (w % stride)  # right
    target_h = h + delta_h
    target_w = w + delta_w
    left = int((target_w - w) / 2.0)
    top = int((target_h - h) / 2.0)
    ltrb = (
        left,
        top,
        delta_w - left,
        delta_h - top,
    )  # 缁欏嚭宸︼紝涓婏紝鍙筹紝涓嬪垎鍒殑pad闀垮害锛屽悗闈orchvision.transforms.functional.pad灏嗕細璇嗗埆鍑虹粰浜4涓猵ad鍙傛暟锛屼粠鑰屽紑濮媝ad
    # ltrb: left, top, right, below

    pad = 4 * [None]
    pad[0] = ltrb[1]  # up
    pad[1] = ltrb[0]  # left
    pad[2] = ltrb[3]  # down
    pad[3] = ltrb[2]  # right

    img_padded = image
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    # 娉ㄦ剰! concatenate 涓や釜鏁扮粍鐨勯『搴忓緢閲嶈
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)
    return img_padded,  pad


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Examples::
        smoothing = GaussianSmoothing(3, 5, 1)
        input = torch.rand(1, 3, 100, 100)
        # 浣跨敤鐨勬槸闀滈潰瀵圭О鐨刾adding绛栫暐锛岃繖鏍锋洿鍚堢悊锛岄伩鍏嶇‖鎬ф坊鍔燾onstant杈圭紭瀵艰嚧杈圭紭鍍忕礌婊ゆ尝寮傚父
        input = F.pad(input, (2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.npu(), groups=self.groups)


def keypoint_heatmap_nms(heat, kernel=3, thre=0.1):
    # keypoint NMS on heatmap (score map)
    pad = (kernel - 1) // 2
    pad_heat = F.pad(heat, (pad, pad, pad, pad), mode='reflect')
    hmax = F.max_pool2d(pad_heat, (kernel, kernel), stride=1, padding=0)
    keep = (hmax == heat).float() * (heat >= thre).float()
    return heat * keep


def refine_centroid(scorefmp, anchor, radius):  # TODO: 娣诲姞keypoint type鐨剆igma瀵箂core鍋氬綊涓鍖栵紝鎶樺悎鎴1.x鍊嶏紝闃叉score琛板噺
    """
    Refine the centroid coordinate. It dose not affect the results after testing.
    :param scorefmp: 2-D numpy array, original regressed score map
    :param anchor: python tuple, (x,y) coordinates
    :param radius: int, range of considered scores
    :return: refined anchor, refined score
    """

    x_c, y_c = anchor
    x_min = x_c - radius
    x_max = x_c + radius + 1
    y_min = y_c - radius
    y_max = y_c + radius + 1

    if y_max > scorefmp.shape[0] or y_min < 0 or x_max > scorefmp.shape[1] or x_min < 0:
        return anchor + (scorefmp[y_c, x_c], )

    score_box = scorefmp[y_min:y_max, x_min:x_max]
    x_grid, y_grid = np.mgrid[-radius:radius+1, -radius:radius+1]
    offset_x = (score_box * x_grid).sum() / score_box.sum()
    offset_y = (score_box * y_grid).sum() / score_box.sum()
    x_refine = x_c + offset_x  # int(np.rint(x_c + offset_x))
    y_refine = y_c + offset_y  #int(np.rint(y_c + offset_y))
    refined_anchor = (x_refine, y_refine)
    return refined_anchor + (score_box.mean(),)


def set_bn_eval_fp32(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval().half()

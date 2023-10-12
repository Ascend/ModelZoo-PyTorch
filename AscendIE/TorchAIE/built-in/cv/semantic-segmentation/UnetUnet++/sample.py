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

import time
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch_aie
from torch_aie import _enums

from model import UNet, NestedUNet


def processing(img_path, msk_path, bs, default_size=(512, 512)):
    img = Image.open(img_path)
    img.resize(default_size, resample=Image.BICUBIC)
    img = np.asarray(img)
    img = img.transpose(2, 0, 1)
    if (img > 1).any():
        img = img / 255
    img = img.astype(np.float32)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.expand(bs, *img_tensor)

    msk = Image.open(msk_path)
    msk.resize(default_size, resample=Image.BICUBIC)
    msk = np.asarray(msk)
    mask_tensor = torch.as_tensor(msk.copy()).long().contiguous()
    mask_tensor = img_tensor.expand(bs, *mask_tensor)

    return img_tensor, mask_tensor


def init_model(model_path, model_type="unet"):
    if model_type == "unet" :
        model_init = UNet(3, 2)
        model_init.load_state_dict(torch.load(model_path, map_location="cpu"))
        model_init.eval()
        return model_init
    else :
        model_init = NestedUNet(3, 2)
        model_init.load_state_dict(torch.load(model_path, map_location="cpu"))
        model_init.eval()
        return model_init


def compile_model(model_compiled, data, data_info):
    trace_model = torch.jit.trace(model_compiled, data)
    pt_model = torch_aie.compile(trace_model,
                                 inputs=data_info,
                                 precision_policy=_enums.PrecisionPolicy.FP16,
                                 allow_tensor_replace_int=True,
                                 soc_version="Ascend310P3")
    return pt_model


def dice_coeff(pred, target, reduce_batch_first=False, eps=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (pred * target).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + eps) / (sets_sum + eps)
    return dice.mean()


def compute_score(mask_pred, mask_true, classes):
    mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
    mask_pred = F.one_hot(mask_pred.argmax(dim=1), classes).permute(0, 3, 1, 2).float()
    # compute the Dice score, ignoring background
    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    print(f"The score is {dice_coeff:0.3f}")


def compute_fps(model_eval, data, loop_counter, warm_counter):
    loops = loop_counter
    while warm_counter:
        _ = model_eval(data)
        warm_counter -= 1

    t0 = time.time()
    while loops:
        _ = model_eval(data)
        loops -= 1
    time_cost = time.time() - t0

    print(f"fps: {loop_counter} * {data.shape[0]} / {time_cost : .3f} samples/s")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_path',
        type=str,
        help="path of demo image"
    )

    parser.add_argument(
        '--mask_path',
        type=str,
        help="path of demo mask"
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="unet",
        help="only support unet/unetpp"
    )

    parser.add_argument(
        '--pth',
        type=str,
        help="path of saved model"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="batch size, default is 1"
    )

    parser.add_argument(
        '--num_class',
        type=int,
        default=2,
        help="num of classes, default is 2"
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help="device id, default is 0"
    )

    parser.add_argument(
        '--loop',
        type=int,
        default=100,
        help="infer times for computing fps, default is 100"
    )

    parser.add_argument(
        '--warm_counter',
        type=int,
        default=10,
        help="loop count before infer, default is 10"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opts = parse_args()
    print(opts)

    input_info = [torch_aie.Input(shape=(1, 3, 512, 512))]
    net = init_model(opts.pth, opts.model_name)
    image, mask = processing(opts.image_path, opts.mask_path, opts.batch_size)
    compiled_model = compile_model(net, image, input_info)
    jit_result = net(image)
    aie_result = compiled_model(image)
    jit_dice_score = compute_score(jit_result, mask, opts.num_class)
    aie_dice_score = compute_score(aie_result, mask, opts.num_class)
    print(f"jit infer score: {jit_dice_score}, aie infer score: {aie_dice_score}")

    cosine_similarity = torch.cosine_similarity(jit_result, aie_result, 0, 1e-6)
    print(f"cosine similarity between jit result and aie result is: {cosine_similarity}")

    compute_fps(compiled_model, image, opts.loop, opts.warm_counter)



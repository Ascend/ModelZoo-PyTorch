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
import os
import glob
import numpy as np
import onnxruntime
import torch
from PIL import Image
from torchvision import transforms
import parse
import PIL.Image as pil
import matplotlib.pyplot as plt


def make_power(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)


def preprocess(PIL_img, image_shape):
    process = transforms.Compose([
        transforms.Lambda(lambda img: make_power(img, base=4, method=Image.BICUBIC)),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)


def postprocess(img_tensor):
    inv_normalize = transforms.Normalize(
        mean=(-1, -1, -1),
        std=(2.0, 2.0, 2.0))
    to_PIL_image = transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0]).clamp(0, 1))


def bin2img_tensor(bin_src):
    # read bin
    with open(bin_src, 'rb') as f:
        imageBin = f.read()
    # What is stored in the bin file is a half-precision file, so we need to convert
    # the binary to half-precision, and restore the model output shape 1*3*256*256
    img_tensor = torch.tensor(np.reshape(np.frombuffer(imageBin, 'f4'), (1, 3, 256, 256)))
    return img_tensor


def main():
    opt = parse.parse_args().initialize()
    if (os.path.exists(opt.bin2img_fie) == False):
        os.makedirs(opt.bin2img_fie)
    npu_bin = glob.glob(opt.npu_bin_file + '*.bin')
    onnxTestImage_path = glob.glob(opt.dataroot + '/testA/*.*')
    model_Ga = onnxruntime.InferenceSession(opt.onnx_path + opt.model_ga_onnx_name)
    cossimis = []
    for i in onnxTestImage_path:
        temp = i.split('/')[4].split('.')[0]
        bin_name = temp + '_1.bin'
        bin_path = opt.npu_bin_file + bin_name
        check = os.path.exists(bin_path)
        if check == True:
            b2imtensor = bin2img_tensor(bin_path)
            pil_image = pil.open(i).convert('RGB')
            tensorData = preprocess(pil_image, 256)
            outputs = model_Ga.run(['maps'], {'img_sat_maps': tensorData.numpy()})
            outputs = torch.tensor(outputs[0])
            cosSimi = torch.mean(torch.cosine_similarity(outputs, b2imtensor))
            cossimis.append(cosSimi.numpy())
    print('average cosine_similarity:')
    print(np.mean(cossimis))
    plt.plot(cossimis)
    plt.xlabel("samples")
    plt.ylabel("cosine_similarity")
    plt.savefig('cosine_similarity.jpg')
    plt.show()


if __name__ == '__main__':
    main()

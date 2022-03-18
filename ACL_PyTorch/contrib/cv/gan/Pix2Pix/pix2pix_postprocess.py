# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

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
    
    if (os.path.exists(bin2img_file) == False):
        os.makedirs(bin2img_file)
    print(npu_bin_file + '*.bin')
    npu_bin_list = glob.glob(npu_bin_file + '*.bin') # 获取指定目录下的所有bin文件
    print(npu_bin_list)
    # onnxTestImage_path = glob.glob(dataroot + '/testA/*.*')
    # model_Ga = onnxruntime.InferenceSession(onnx_path + model_pix2pix_onnx_name)
    # cossimis = []
    for npu_bin in npu_bin_list:
        b2imtensor = bin2img_tensor(npu_bin)
        image_numpy = tensor2im(b2imtensor)
        image_name = npu_bin.split('/')[-1].split('.')[0]+ '.jpg'
        print(image_name)
        image_save_path = bin2img_file+image_name 
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_save_path)
    # print('average cosine_similarity:')
    # print(np.mean(cossimis))
    # plt.plot(cossimis)
    # plt.xlabel("samples")
    # plt.ylabel("cosine_similarity")
    # plt.savefig('cosine_similarity.jpg')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin2img_file', help='bin2img_file')
    parser.add_argument('--npu_bin_file', help='npu_bin_file')
    args = parser.parse_args()


    # bin2img_file = './result/bin2img_bs16/'
    # npu_bin_file = './result/dumpOutput_device0_bs16/'

    bin2img_file = args.bin2img_file
    npu_bin_file = args.npu_bin_file

    # dataroot = './datasets/facades'
    
    main()

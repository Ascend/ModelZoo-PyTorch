# Copyright 2021 Huawei Technologies Co., Ltd
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
from os import listdir
from os.path import join
from PIL import Image
import torch
from math import log10
import onnxruntime
import numpy as np

import pytorch_ssim
import torchvision.utils as utils
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage, CenterCrop


class ONNXModel():
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        segmap = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return segmap


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 通过后缀检查是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def get_right_size_img(lr_filenames, hr_filenames, width, height):
    lr_img = []
    for img_path in lr_filenames:
        image = Image.open(img_path)
        w, h = image.size
        if w == width and h == height:
            lr_img.append(img_path)
    hr_img = []
    for img_path in hr_filenames:
        image = Image.open(img_path)
        w, h = image.size
        if w == 2 * width and h == 2 * height:
            hr_img.append(img_path)
    if lr_img and hr_img:
        return lr_img, hr_img
    else:
        print('未找到与目标尺寸对应的 lr 和 hr 图片。')


def test_data_process(lr_img_path_list, hr_img_path_list):
    results = []
    for i in range(len(lr_img_path_list)):
        img_name = lr_img_path_list[i].split(os.sep)[-1]
        lr_image = Image.open(lr_img_path_list[i]).convert('RGB')
        w, h = lr_image.size
        hr_image = Image.open(hr_img_path_list[i]).convert('RGB')
        hr_scale = Resize((2 * h, 2 * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)

        results.append([img_name,
                        ToTensor()(lr_image).unsqueeze(0),
                        ToTensor()(hr_restore_img).unsqueeze(0),
                        ToTensor()(hr_image).unsqueeze(0)])
    return results


# 用于输出图像的转换
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),  # 把图像调整到400标准格式
        CenterCrop(400),
        ToTensor()
    ])

def load_bin_img(img_name, path):
    bin_name = img_name.split('.')[0]+'.bin'
    bin_img = np.fromfile(join(path, bin_name), dtype='float32')
    return bin_img


if __name__ == '__main__':
    models_file = 'onnx_models'
    pre_file_path = '../set5'
    output_path = './infer_onnx_res'
    foward_process = 'foward_process'
    # 设置保存路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = join(pre_file_path, 'data')
    target_path = join(pre_file_path, 'target')
    # 读取待测试图和目标图
    lr_filenames = [join(data_path, x) for x in listdir(data_path) if is_image_file(x)]
    hr_filenames = [join(target_path, x) for x in listdir(target_path) if is_image_file(x)]
    print(lr_filenames)
    print(hr_filenames)
    # lr_img_path_list, hr_img_path_list = get_right_size_img(lr_filenames, hr_filenames, 140, 140)
    test_data = test_data_process(lr_filenames, hr_filenames)

    results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
               'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}
    load_bin = True

    for i, data in enumerate(test_data):
        image_name, lr_image, hr_restore_img, hr_image = data
        if load_bin:
            bin_img = load_bin_img(image_name,foward_process)
            bin_img = bin_img.reshape(lr_image.shape)
            # bin_img = torch.tensor(bin_img)
        h,w = lr_image.shape[2:]
        print(lr_image.shape)
        # 生成模型
        model_name = 'srgan_{}_{}.onnx'.format(h,w)
        netG = ONNXModel(os.path.join(models_file,model_name))
        if load_bin:
            lr_image = bin_img
        else:
            lr_image = to_numpy(lr_image)
        sr_image = netG.forward(lr_image)
        sr_image = torch.tensor(sr_image)
        sr_image = torch.squeeze(sr_image, 0)
        # if sr_image.shape != hr_image.shape:
        #     sr_image = sr_image.transpose(2,3)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, output_path + '/' + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)
        print(f'图片 {image_name} 的生成结果--> PSNR: {psnr:.4f}  SSIM: {ssim:.4f}')
        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)
    print('推理完毕！')

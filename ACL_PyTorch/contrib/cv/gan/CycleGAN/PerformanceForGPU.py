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
import os
import time
import torchvision.transforms as transforms
from PIL import Image
import torch.onnx
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from parse import parse_args
import numpy as np
from CycleGAN_NetLoad import load_networks


def make_power(img, base):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img


def preprocess(image_shape):
    process = transforms.Compose([
        transforms.Lambda(lambda img: make_power(img, base=4)),
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return process


def postprocess(img_tensor):
    inv_normalize = transforms.Normalize(
        mean=(-1, -1, -1),
        std=(2.0, 2.0, 2.0))
    to_PIL_image = transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0]).clamp(0, 1))


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n" +
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def main():
    paser = parse_args(True, True)
    opt = paser.initialize()
    lnetworks = load_networks(opt)
    bachsize = opt.batch_size
    # whether to use fp16 to farword
    half_data_model = True
    transform = preprocess((256, 256))
    model_Ga, model_Gb = lnetworks.get_networks(opt.model_ga_path, opt.model_gb_path)
    device_cuda = torch.device("cuda:%s" % (str(opt.pu_ids)))
    model_Ga = model_Ga.to(device_cuda)
    if (half_data_model):
        model_Ga = model_Ga.half()
    datasets = ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=bachsize, shuffle=True, num_workers=4)
    filename = opt.gpuPerformance + 'GPU_perf_of_cycle_gan-b0_bs' + str(bachsize) + '_in_device_' + str(
        opt.pu_ids) + '.txt'
    f = None
    if (os.path.exists(opt.gpuPerformance) == False):
        os.mkdir(opt.gpuPerformance)
        f = open(filename, mode='w')
    else:
        f = open(filename, mode='w')
    timelist = []
    for i, data in enumerate(dataloader):
        start_time = time.time()
        data = data.to(device_cuda)
        if (half_data_model):
            data = data.half()
        model_Ga.forward(data)
        end_time = time.time()
        if (i > 10):
            timelist.append((end_time - start_time) * 1000)
    a_time = time.asctime(time.localtime(time.time()))
    timelist = np.array(timelist)
    mintime = timelist.argmin()
    maxtime = timelist.argmax()
    meantime = np.mean(timelist)
    mediantime = np.median(timelist)
    alltime = np.sum(timelist) / 1000
    message = '''
     [%s],[I]  GPU Compute
     [%s],[I]  min:%.5f ms
     [%s],[I]  max:%.5f ms
     [%s],[I]  mean:%.5f ms
     [%s],[I]  median:%.5f ms
     [%s],[I]  total compute time:%.5f s
     [%s],[I]  CardFPS:1000/(%f/%f)=%.2f fps
    ''' % (a_time, \
           a_time, mintime, \
           a_time, maxtime, \
           a_time, meantime, \
           a_time, mediantime, \
           a_time, alltime, \
           a_time, meantime, bachsize, 1000 / (meantime / bachsize))

    print(message)
    f.write(message)
    f.close()


if __name__ == '__main__':
    main()

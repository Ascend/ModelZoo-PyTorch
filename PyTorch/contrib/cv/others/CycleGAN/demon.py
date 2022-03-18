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
import torchvision.transforms as transforms
from PIL import Image
import torch.onnx
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from parse import parse_args
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
    def __init__(self, root, transform=None, return_paths=True,
                 loader=default_loader):
        imgs = make_dataset(root + '/testA')
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


def deal_tensor(datas, outputs):
    res_img = postprocess(datas)
    res_gimg = postprocess(outputs)


def main():
    paser = parse_args(True, True)
    opt = paser.initialize()
    htmlres = ''

    pathroot = './result/'
    images_name = 'img'
    if (os.path.exists(pathroot + images_name) == False):
        os.makedirs(pathroot + images_name)
    f = open(pathroot + 'index.html', 'w')
    lnetworks = load_networks(opt)
    bachsize = opt.batch_size
    loc_cpu = 'cpu'
    loc = 'npu:1'
    transform = preprocess((256, 256))
    model_Ga, _ = lnetworks.get_networks(opt.model_ga_path, opt.model_gb_path)
    model_Ga.eval()
    datasets = ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=bachsize, shuffle=True, num_workers=4)

    count = 0
    for i, (x, x_path) in enumerate(dataloader):
        count += 1
        if (count > 10):
            break
        temp = str(x_path).split('/')
        img_name = temp[4].split(',')[0].split('\'')[0]
        src_real = temp[3]
        src_g = temp[3] + 'G'
        if (os.path.exists(pathroot + images_name + '/' + src_real) == False):
            os.makedirs(pathroot + images_name + '/' + src_real)
        if (os.path.exists(pathroot + images_name + '/' + src_g) == False):
            os.makedirs(pathroot + images_name + '/' + src_g)
        x1 = postprocess(x)
        realsrc = images_name + '/' + src_real + '/' + img_name
        fakesrc = images_name + '/' + src_g + '/' + img_name
        y = model_Ga(x.to(loc))
        y = postprocess(y.to(loc_cpu))
        x1.save(pathroot + realsrc)
        y.save(pathroot + fakesrc)
        htmlres += '''
              <div class='img_box'>
              <div class='img'>
              <p>%s</p>
               <img  src=%s />
              </div>
              <div class='img'>
              <p>%s</p>
               <img  src=%s />
              </div>
              </div>
              ''' % (img_name.split('.')[0], realsrc, img_name.split('.')[0] + '_fake', fakesrc)

    htmlshow = """<html>
        <head></head>
        <style type='text/css'>
           .img_box{
           display: flex;
           width:100%%;
           }
           .img{
           display:inline;
           float:left;
           margin-left:2px;
           }
        
        </style>
          %s
        </body>
        </html>""" % (htmlres)
    f.write(htmlshow)
    f.close()


if __name__ == '__main__':
    main()

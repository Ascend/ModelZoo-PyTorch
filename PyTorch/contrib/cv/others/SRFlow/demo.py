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


import glob
import sys
import os
os.chdir('./code')
sys.path.append(r'./')
from collections import OrderedDict
from natsort import natsort
import options.options as option
from skimage.metrics import peak_signal_noise_ratio as psnr
from models import create_model
import torch
from utils.util import opt_get
from utils import util
import numpy as np
import pandas as pd
import cv2
import PIL.Image as pil_image
import torchvision.transforms as transforms


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
    np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def main():
    util.set_random_seed(0)
    conf_path = './confs/SRFlow_DF2K_8X.yml'
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)

    lr_dir = opt['dataroot_LR']
    hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results', conf)
    print(f"Out dir: {test_dir}")

    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    scale = opt['scale']

    pad_factor = 2

    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
        hr = imread(hr_path)
        pil_lr = pil_image.fromarray(np.uint8(lr))
        lr = transforms.Pad(padding=(0, 0, 256 - pil_lr.size[0], 256 - pil_lr.size[1]), padding_mode='edge')(
            pil_lr)
        lr = np.array(lr)
        # Pad image to be % 2
        h, w, c = lr.shape
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr_t = t(lr)

        heat = opt['heat']

        if df is not None and len(df[(df['heat'] == heat) & (df['name'] == idx_test)]) == 1:
            continue

        sr_t = model.get_sr(lq=lr_t, heat=heat)

        sr = rgb(torch.clamp(sr_t, 0, 1))
        sr = sr[:pil_lr.size[1] * scale, :pil_lr.size[0] * scale]
        path_out_sr = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), "{:06d}.png".format(idx_test))
        imwrite(path_out_sr, sr)

        meas = OrderedDict(conf=conf, heat=heat, name=idx_test)
        meas['PSNR'] = psnr(sr, hr)

        str_out = format_measurements(meas)
        print(str_out)

        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])

        df.to_csv(path_out_measures + "_", index=False)
        os.rename(path_out_measures + "_", path_out_measures)

    df.to_csv(path_out_measures, index=False)
    os.rename(path_out_measures, path_out_measures_final)

    str_out = format_measurements(df.mean())
    print(f"Results in: {path_out_measures_final}")
    print('Mean: ' + str_out)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()

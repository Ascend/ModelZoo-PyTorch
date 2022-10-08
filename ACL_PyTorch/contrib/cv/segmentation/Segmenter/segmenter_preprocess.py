# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from pathlib import Path

import tqdm
import yaml
import numpy as np

import torch
import segm.utils.torch as ptu
from segm.data.factory import create_dataset
from segm.model.utils import resize, sliding_window
from segm import config


def preprocess(variant_path, save_path, gt_file_path):

    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"]["cityscapes"]
    normalization = variant["dataset_kwargs"]["normalization"]
    im_size = dataset_cfg.get("im_size", variant["dataset_kwargs"]["image_size"])
    window_size = variant["dataset_kwargs"]["crop_size"]
    window_stride = variant["dataset_kwargs"]["crop_size"] - 32
    C=19

    dataset_kwargs = dict(
        dataset="cityscapes",
        image_size=im_size,
        crop_size=im_size,
        patch_size=16,
        batch_size=1,
        num_workers=10,
        split="val",
        normalization=normalization,
        crop=False,
        rep_aug=False,
    )

    db = create_dataset(dataset_kwargs)
    seg_gt_maps = db.dataset.get_gt_seg_maps()
    # save groundtruth mapping
    with open(gt_file_path, 'w', encoding='utf-8') as f:
        for img_info in db.dataset.dataset.img_infos:
            gt_path = Path(db.dataset.dataset.ann_dir) / img_info["ann"]["seg_map"]
            line = f"{Path(img_info['filename']).name}\t{gt_path.__str__()}\n"
            f.write(line)

    im_size = dataset_kwargs["image_size"]
    cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls

    for batch in tqdm.tqdm(db, desc='Processing'):
        colors = batch["colors"]
        ims = batch["im"]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        seg_map = torch.zeros((C, ori_shape[0], ori_shape[1]), device=ptu.device)
        for im, im_metas in zip(ims, ims_metas):
            im = im.to(ptu.device)
            im = resize(im, window_size)
            flip = im_metas["flip"]
            windows = sliding_window(im, flip, window_size, window_stride)
            crops = torch.stack(windows.pop("crop"))[:, 0]
            B = len(crops)
            WB = 1
            seg_maps = torch.zeros((B, C, window_size, window_size), device=im.device)
            cnt = 0
            for i in range(0, B, WB):
                input_tensor = crops[i : i + WB]
                ori_image_name = im_metas['ori_filename'][0]
                img = np.array(input_tensor).astype(np.float32)
                filename = Path(ori_image_name).stem + "_" + str(cnt)
                img.tofile(os.path.join(save_path, filename + ".bin"))
                cnt += 1


def main():

    import argparse
    parser = argparse.ArgumentParser('image convert to binary file.')
    parser.add_argument('--cfg-path', type=str, required=True,
        help='path to model config file.')
    parser.add_argument('--data-root', type=str, required=True,
        help='path to parent directory of cityscapes dataset.')
    parser.add_argument('--bin-dir', type=str, required=True,
        help='directory to save binary file.')
    parser.add_argument('--gt-path', type=str, required=True,
        help='save the mapping of (srcImage, labelImage)')
    args = parser.parse_args()

    os.environ['DATASET'] = args.data_root
    if not Path(args.bin_dir).is_dir():
        Path(args.bin_dir).mkdir(parents=True, exist_ok=True)
    preprocess(args.cfg_path, args.bin_dir, args.gt_path)


if __name__ == '__main__':
    main()

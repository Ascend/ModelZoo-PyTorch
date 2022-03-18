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
'''
  merge eight 768X768 img into one 1024X2048 img
'''
import os
import sys

import numpy as np
from mmseg.datasets import build_dataloader, build_dataset
import torch.nn.functional as F
import warnings
import torch
from mmcv.utils import Config
import mmcv
import json

def gen_data_loader(distributed, cfg):

    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
    return val_dataset, val_dataloader

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        'When align_corners={}, '.format(align_corners) + \
                        'the output would more aligned if ' + \
                        'input size {} is `x+1` and '.format((input_h, input_w)) + \
                        'out size {} is `nx+1`'.format((output_h, output_w))) 
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def merge768(data_loader, om_output_dir, output_dir, test_cfg, rescale=True, num_classes=19, align_corners=True):
        """
            根据val数据集名字，将8张bin图片合并
        """ 
        for i, data in enumerate(data_loader):
            
            file_name = data['img_metas'][0].data[0][0]['ori_filename']
            file_id = file_name.split('.')[0].split('/')[-1]
            img = data['img'][0] # tensor 1,3,1025, 2049
            img_meta = data['img_metas'][0]

            h_stride, w_stride = test_cfg.stride
            h_crop, w_crop = test_cfg.crop_size
            batch_size, _, h_img, w_img = img.size()
            num_classes = num_classes
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
            count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
            index = 0
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    
                    cur_file_name =  "{}[{}]_1.bin".format(file_id, index)
                    crop_seg_logit = np.fromfile(os.path.join(om_output_dir, cur_file_name), dtype='float32')
                    crop_seg_logit = np.reshape(crop_seg_logit, (1, 19, 768, 768))
                    crop_seg_logit = torch.from_numpy(crop_seg_logit)
                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
                    index += 1 
            assert (count_mat == 0).sum() == 0

            preds = preds / count_mat
            if rescale:
                preds = resize(
                    preds,
                    size=img_meta.data[0][0]['ori_shape'][:2],
                    mode='bilinear',
                    align_corners=align_corners,
                    warning=False)
            output = F.softmax(preds, dim=1)
            seg_pred = output.argmax(dim=1)
            output_path = os.path.join(output_dir, "/".join(file_name.split('/')[:-1]))
            if not os.path.exists(output_path):
                    os.makedirs(output_path)
            seg_pred = np.array(seg_pred).astype(np.float32)
            seg_pred.tofile(os.path.join(output_dir, file_name.split('.')[0] + '.bin'))



def evaluate(dataset, pred_dir, metric='mIoU'):

    preds = []
    for i, img in enumerate(dataset.img_infos):
        img_pred_name = img['filename']
        pred = np.fromfile(os.path.join(pred_dir, img_pred_name.split(".")[0]+".bin"), dtype='float32')
        pred = np.reshape(pred, (1024, 2048))
        preds.append(pred)
    return dataset.evaluate(preds, metric)

if __name__ == '__main__':
    model_config_file   = sys.argv[1]
    om_output_dir       = sys.argv[2]    # pred文件目录
    output_dir          = sys.argv[3]    # 整合后pred文件目录  
    res_file_name       = sys.argv[4]
    cfg = Config.fromfile(model_config_file)
    dataset, dataloader = gen_data_loader(False, cfg)
    print("正在整合生成图片文件,please wait for a mement.")
    merge768(dataloader, om_output_dir, output_dir, cfg.test_cfg)
    print("正在对图片进行评估,please wait for a moment.")
    pred_dir = output_dir
    eval_cfg = cfg.get('evaluation', {})
    res = evaluate(dataset, pred_dir)
    with open(res_file_name + ".json", 'w') as f:
        json.dump(res, f)
    
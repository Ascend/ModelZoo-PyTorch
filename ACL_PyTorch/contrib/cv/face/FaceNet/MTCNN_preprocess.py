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
import sys
import json
import torch
import argparse
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
sys.path.append('./models')
from mtcnn import PNet, RNet, ONet
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import imresample, generateBoundingBox, batched_nms, rerec, pad, bbreg, batched_nms_numpy


NET_MAP = {
    'pnet': PNet,
    'rnet': RNet,
    'onet': ONet
}

###################################################################################
# basic function #
###################################################################################

def build_dataset(config):
    orig_img_ds = datasets.ImageFolder(config.data_dir, transform=None)
    orig_img_ds.samples = [(p, p)for p, _ in orig_img_ds.samples]
    def collate_fn(x):
        out_x, out_y = [], []
        for xx, yy in x:
            out_x.append(xx)
            out_y.append(yy)
        return out_x, out_y
    loader = DataLoader(
        orig_img_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )
    return loader


def dump_to_json(content, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(content, f)


def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)


###################################################################################
# main class #
###################################################################################
class MTCNNPreprocessor():
    def __init__(self, config):
        self.net_name = config.net
        self.net = NET_MAP[self.net_name](config)
        self.threshold = [0.6, 0.7, 0.7]
        self.data_device = torch.device('cpu')

    def pnet_process(self, imgs):
        if self.net_name != 'pnet':
            raise ValueError('Pnet process not support for {} !'.format(self.net))

        factor = 0.709
        minsize = 20

        imgs = imgs.permute(0, 3, 1, 2).type(torch.float32)
        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        m = 12.0 / minsize
        minl = min(h, w)
        minl = minl * m

        scale_i = m
        scales = []
        while minl >= 12:
            scales.append(scale_i)
            scale_i = scale_i * factor
            minl = minl * factor
        # First stage
        boxes = []
        image_inds = []
        scale_picks = []
        all_i = 0
        offset = 0
        for scale in scales:
            im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
            im_data = (im_data - 127.5) * 0.0078125
            reg, probs = self.net.forward(im_data.cpu().numpy())
            reg = torch.from_numpy(reg)
            probs = torch.from_numpy(probs)
            boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, self.threshold[0])
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)
            pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]
        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0)
        scale_picks = torch.cat(scale_picks, dim=0)
        # NMS within each scale + image
        boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]
        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick]
        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = rerec(boxes)
        return boxes, image_inds

    def rnet_process(self, imgs, boxes, image_inds):
        if self.net_name != 'rnet':
            raise ValueError('Rnet process not support for {} !'.format(self.net))
        imgs = imgs.permute(0, 3, 1, 2).type(torch.float32)
        h, w = imgs.shape[2:4]
        y, ey, x, ex = pad(boxes, w, h)
        if len(boxes) > 0:
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (24, 24)))
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            out = self.net.forward(im_data.cpu().numpy())
            out = [torch.from_numpy(o) for o in out]
            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > self.threshold[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)
            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)
        return boxes, image_inds

    def onet_process(self, imgs, boxes, image_inds):
        if self.net_name != 'onet':
            raise ValueError('Onet process not support for {} !'.format(self.net))
        imgs = imgs.permute(0, 3, 1, 2).type(torch.float32)
        h, w = imgs.shape[2:4]
        points = torch.zeros(0, 5, 2, device=self.data_device)
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h)
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (48, 48)))
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            out = self.net.forward(im_data.cpu().numpy())
            out = [torch.from_numpy(o) for o in out]
            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            points = out1
            ipass = score > self.threshold[2]
            points = points[:, ipass]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)
            w_i = boxes[:, 2] - boxes[:, 0] + 1
            h_i = boxes[:, 3] - boxes[:, 1] + 1
            points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
            points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
            points = torch.stack((points_x, points_y)).permute(2, 1, 0)
            boxes = bbreg(boxes, mv)
            # NMS within each image using "Min" strategy
            # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
            boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()
        image_inds = image_inds.cpu()
        batch_boxes = []
        batch_points = []
        for b_i in range(config.batch_size):
            b_i_inds = np.where(image_inds == b_i)
            batch_boxes.append(boxes[b_i_inds].copy())
            batch_points.append(points[b_i_inds].copy())
        batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)
        return batch_boxes, batch_points


###################################################################################
# main function #
###################################################################################
def process_pnet(config):
    loader = build_dataset(config)
    processor = MTCNNPreprocessor(config)
    out_json = {}
    for idx, (xs, b_paths) in tqdm(enumerate(loader), total=len(loader)):
        imgs = np.stack([np.uint8(x) for x in xs])
        imgs = torch.as_tensor(imgs.copy(), device=torch.device('cpu'))
        boxes, image_inds = processor.pnet_process(imgs)
        out_json[str(idx)] = {
        'boxes': boxes.tolist(),
        'image_inds': image_inds.tolist()
        }
    save_path = os.path.join(config.output_path, 'pnet.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_to_json(out_json, save_path)


def process_rnet(config):
    loader = build_dataset(config)
    processor = MTCNNPreprocessor(config)
    out_json = {}
    pnet_data = load_json(config.input_path)
    for idx, (xs, b_paths) in tqdm(enumerate(loader), total=len(loader)):
        imgs = np.stack([np.uint8(x) for x in xs])
        imgs = torch.as_tensor(imgs.copy(), device=torch.device('cpu'))
        boxes = torch.from_numpy(np.array(pnet_data[str(idx)]['boxes']))
        image_inds = torch.from_numpy(np.array(pnet_data[str(idx)]['image_inds']))
        boxes, image_inds = processor.rnet_process(imgs, boxes, image_inds)
        out_json[str(idx)] = {
            'boxes': boxes.tolist(),
            'image_inds': image_inds.tolist()
        }
    save_path = os.path.join(config.output_path, 'rnet.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_to_json(out_json, save_path)


def process_onet(config):
    data_dir = config.data_dir
    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]
    loader = build_dataset(config)
    processor = MTCNNPreprocessor(config)
    pnet_data = load_json(config.input_path)
    crop_paths = []
    for idx, (xs, b_paths) in tqdm(enumerate(loader), total=len(loader)):
        imgs = np.stack([np.uint8(x) for x in xs])
        imgs = torch.as_tensor(imgs.copy(), device=torch.device('cpu'))
        boxes = torch.from_numpy(np.array(pnet_data[str(idx)]['boxes']))
        image_inds = torch.from_numpy(np.array(pnet_data[str(idx)]['image_inds']))
        batch_boxes, batch_points = processor.onet_process(imgs, boxes, image_inds)
        # save crop imgs
        save_paths = [p.replace(data_dir, data_dir + '_split_om_cropped_{}'.format(config.batch_size)) for p in b_paths]
        save_crop_imgs(batch_boxes, batch_points, xs, save_paths)
        crop_paths.extend(save_paths)
    save_path = os.path.join(config.output_path, 'onet.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_to_json(crop_paths, save_path)


def save_crop_imgs(batch_boxes, batch_points, img, save_path):
    mtcnn = MTCNN(
        image_size=160, margin=14, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        selection_method='center_weighted_size'
    )
    boxes, probs, points = [], [], []
    for box, point in zip(batch_boxes, batch_points):
        box = np.array(box)
        point = np.array(point)
        if len(box) == 0:
            boxes.append(None)
            probs.append([None])
            points.append(None)
        elif mtcnn.select_largest:
            box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
            box = box[box_order]
            point = point[box_order]
            boxes.append(box[:, :4])
            probs.append(box[:, 4])
            points.append(point)
        else:
            boxes.append(box[:, :4])
            probs.append(box[:, 4])
            points.append(point)
    batch_boxes = np.array(boxes)
    batch_probs = np.array(probs)
    batch_points = np.array(points)

    batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
        batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
    )
    # Extract faces
    faces = mtcnn.extract(img, batch_boxes, save_path)
    return faces


def build_config(arg):
    pnet_config = {
        'net': 'pnet',
        'device_id': arg.device_id,
        'output_path': './data/output/split_bs' + str(arg.batch_size)  + '/',
        'model_path': './weights/PNet_dynamic.om',
        'data_dir': arg.data_dir,
        'num_workers': 8,
        'batch_size': arg.batch_size
    }
    rnet_config = {
        'net': 'rnet',
        'device_id': arg.device_id,
        'input_path': './data/output/split_bs' + str(arg.batch_size)  + '/pnet.json',
        'output_path': './data/output/split_bs' + str(arg.batch_size) + '/',
        'model_path': './weights/RNet_dynamic.om',
        'data_dir': arg.data_dir,
        'num_workers': 8,
        'batch_size': arg.batch_size
    }
    onet_config = {
        'net': 'onet',
        'device_id': arg.device_id,
        'input_path': './data/output/split_bs' + str(arg.batch_size)  + '/rnet.json',
        'output_path': './data/output/split_bs' + str(arg.batch_size)  + '/',
        'model_path': './weights/ONet_dynamic.om',
        'data_dir': arg.data_dir,
        'num_workers': 8,
        'batch_size': arg.batch_size
    }
    return EasyDict(pnet_config), EasyDict(rnet_config), EasyDict(onet_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='[PNet/RNet/ONet]')
    parser.add_argument('--data_dir', type=str, help='the absolute files path of lfw dataset')
    parser.add_argument('--batch_size', default=1, type=int, help='[1/16]')
    parser.add_argument('--device_id', default=0, type=int)
    arg = parser.parse_args()
    pnet_config, rnet_config, onet_config = build_config(arg)
    if arg.model == 'Pnet':
        config = pnet_config
        process_pnet(config)
    elif arg.model == 'Rnet':
        config = rnet_config
        process_rnet(config)
    elif arg.model == 'Onet':
        config = onet_config
        process_onet(config)

# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trans

from model import Backbone, l2_norm


def build_model(input_file):
    model = Backbone(num_layers=100, drop_ratio=1, mode='ir_se')

    ckpt = torch.load(input_file, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model.eval()

    return model


def get_image(image_path, transform):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = transform(img).unsqueeze(0)
    return img


def inference(model, image, device):
    image = image.to(device)
    img_flip = torch.flip(image, dims=[3]).to(device)
    emb_batch = l2_norm(model(image) + model(img_flip))
    return emb_batch


def prepare_facebank(facebank_dir, transform, model, device, tta=True):
    """
    facebank_dir：facebank_dir
    transform： image transform
    model： test model
    device
    tta:
    """
    model.eval()
    embeddings = []
    names = ['Unknown']
    for path in facebank_dir.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue

                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(transform(img).to(device).unsqueeze(0))
                            emb_mirror = model(transform(mirror).to(device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(transform(img).to(device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, facebank_dir / 'facebank.pth')
    np.save(facebank_dir / 'names', names)
    return embeddings, names


def build_facebank(facebank_path, facebank_name_path):
    """
    facebank_path：人脸数据库
    facebank_name_path：人脸名称数据库
    """
    embeddings = torch.load(facebank_path, map_location='cpu')
    names = np.load(facebank_name_path)
    return embeddings, names


def check(emb_batch, target_embs, names, threshold=1.54):
    """
    emb_batch：model预测出的 embedding
    target_embs: 人脸数据库
    names：人脸名称数据库
    threshold：由 eval 得到
    """
    diff = emb_batch.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > threshold] = -1  # if no match, set idx to -1
    result, score = min_idx, minimum
    print('*' * 50)
    print('facebank name list: {}'.format(names))
    print('result: {}'.format(result))
    print('name: {}'.format(names[result.item() + 1]))
    print('score: {}'.format(score))
    print('*' * 50)


def prepare_parser():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("--weights_path", help="model weights path", default='./work_space/save/model_ir_se100.pth')
    parser.add_argument("--data_path", help="input image file path", default='./demo_img.jpg')

    parser.add_argument("--check", help="check person in facebank", default=0, type=int)
    parser.add_argument("--update", help="whether perform update the facebank", default=0, type=int)
    parser.add_argument("--facebank_dir", help="facebank dir", default='./data/facebank')
    parser.add_argument("--threshold", help="eval threshold", default=1.54, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = prepare_parser()

    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    device = torch.device('cpu')

    # 构建模型
    model = build_model(args.weights_path)

    # 获取图片
    image = get_image(args.data_path, transform)

    # 开始预测
    emb_batch = inference(model, image, device)
    print('emb_batch: {}'.format(emb_batch.detach().numpy()))

    # 验证人脸库进行验证
    if args.check:
        facebank_dir = Path(args.facebank_dir)
        if args.update:
            # 建立face_bank
            print('start building face_bank, Have a cup of coffee first ? ')
            embeddings, names = prepare_facebank(facebank_dir, transform, model, device, tta=True)
        else:
            facebank_path = facebank_dir / 'facebank.pth'
            facebank_name_path = facebank_dir / 'names.npy'
            if not facebank_path.exists() and not facebank_name_path.exists():
                raise FileExistsError("please update facebank first")
            # 读取face_bank
            print('start loading face_bank')
            embeddings, names = build_facebank(facebank_path, facebank_name_path)

        # 进行对比
        check(emb_batch, embeddings, names, args.threshold)

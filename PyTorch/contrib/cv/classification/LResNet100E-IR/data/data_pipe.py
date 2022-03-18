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
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader, DistributedSampler
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
from tqdm import tqdm

# arm 架构不容易安装mxnet，容易发生报错
try:
    import mxnet as mx
except:
    Warning("If it is ARM architecture, Please download whole Dataset")

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    else:
        ds, class_num = get_train_dataset(conf.emore_folder/conf.data_mode)

    if conf.distributed:
        ds_sample = DistributedSampler(ds, num_replicas=conf.world_size, rank=conf.rank)
        loader = DataLoader(ds,
                            batch_size=conf.batch_size,
                            shuffle=False,
                            pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers,
                            sampler=ds_sample)
    else:
        loader = DataLoader(ds,
                            batch_size=conf.batch_size,
                            shuffle=True,
                            pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers)
    return loader, class_num

def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img_np_arr = np.frombuffer(_bin, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

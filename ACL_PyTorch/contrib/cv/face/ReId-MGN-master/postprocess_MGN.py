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
# limitations under the License.jj

import os
import sys
import numpy as np
import torch
import argparse
from scipy.spatial.distance import cdist
from tqdm import tqdm
sys.path.append('./MGN')
from MGN.data import Data
from MGN.utils.metrics import mean_ap, cmc, re_ranking


def save_batch_imgs(save_file_name, dataset_type, loader, need_flip=False):
    ind = 0
    for (inputs, labels) in loader:
        if need_flip == True:
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
        for i in range(len(inputs)):
            img_name =  dataset_type + '/' + "{:0>5d}".format(ind)
            save_path = opt.data_path
            if(opt.data_path[-1] != '/'):
                save_path += '/'
            save_path += save_file_name
            inputs[i].numpy().tofile(save_path + '/' + img_name + '.bin')
            ind += 1


def extract_feature_om(prediction_file_path, prediction_file_path_flip):
    # make the list of files first
    file_names, file_names_flip = [], []
    for file_name in os.listdir(prediction_file_path):
        suffix = file_name.split('_')[-1]
        if suffix == '1.txt':
            file_names.append(file_name)
    file_names.sort()
    print("first 5 txt files: \n",file_names[:10])
    for file_name in os.listdir(prediction_file_path_flip):
        suffix = file_name.split('_')[-1]
        if suffix == '1.txt':
            file_names_flip.append(file_name)
    file_names_flip.sort()
    if len(file_names) != len(file_names_flip):
        print('num of filp features doesnt match that of orig')
    features = torch.FloatTensor()
    for i in range(len(file_names)):
        fea_path   = os.path.join(prediction_file_path, file_names[i])
        fea_path_f = os.path.join(prediction_file_path_flip, file_names_flip[i])
        f1 = torch.from_numpy(np.loadtxt(fea_path, dtype=np.float32))
        f2 = torch.from_numpy(np.loadtxt(fea_path_f, dtype=np.float32))
        ff = f1 + f2
        ff = torch.unsqueeze(ff, 0)
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        if i < 8:
            print(i, "th f1: \n", f1.shape, f1)
            print(i, "th f2: \n", f2.shape, f2)
            print(i, "th ff: \n", ff.shape, ff)
        if i % 100 == 0:
            print("the " + str(i) + "th image file is extracted.")
    return features


class Main():
    def __init__(self, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

    def evaluate_om(self):
        query_prediction_file_path, query_prediction_file_path_flip = './result/q_bin/dumpOutput_device0/', \
                                                                      './result/q_bin_flip/dumpOutput_device0/'
        gallery_prediction_file_path, gallery_prediction_file_path_flip = './result/g_bin/dumpOutput_device0/', \
                                                                          './result/g_bin_flip/dumpOutput_device0/'
        print('extract features, this may take a few minutes')
        qf = extract_feature_om(query_prediction_file_path, query_prediction_file_path_flip).numpy()
        gf = extract_feature_om(gallery_prediction_file_path, gallery_prediction_file_path_flip).numpy()
        print("shape of features, qf: " + str(qf.shape) + "gf: " + str(gf.shape))
        print("arr qf: \n", qf[:10, :10])
        print("arr gf: \n", gf[:10, :10])

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
            return r, m_ap
        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        r, m_ap = rank(dist)
        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        #########################no re rank##########################
        dist = cdist(qf, gf)
        r, m_ap = rank(dist)
        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def save_data(self):
        save_file_name = 'bin_data'
        save_file_name_flip = 'bin_data_flip'
        print('saving images, this may take a few minutes')
        save_batch_imgs(save_file_name, 'q', tqdm(self.query_loader))
        save_batch_imgs(save_file_name, 'g', tqdm(self.test_loader))
        save_batch_imgs(save_file_name_flip, 'q', tqdm(self.query_loader), need_flip=True)
        save_batch_imgs(save_file_name_flip, 'g', tqdm(self.test_loader), need_flip=True)


def parse_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default="Market-1501-v15.09.15",
                        help='path of Market-1501-v15.09.15')
    parser.add_argument('--mode',
                        default='train', choices=['train', 'evaluate', 'evaluate_om', 'save_bin', 'vis'],
                        help='train or evaluate ')
    parser.add_argument('--query_image',
                        default='0001_c1s1_001051_00.jpg',
                        help='path to the image you want to query')
    parser.add_argument("--batchid",
                        default=4,
                        help='the batch for id')
    parser.add_argument("--batchimage",
                        default=4,
                        help='the batch of per id')
    parser.add_argument("--batchtest",
                        default=8,
                        help='the batch size for test')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_func()
    data = Data(opt)
    main = Main(data)
    if opt.mode == 'evaluate_om':
        print('start evaluate om')
        main.evaluate_om()
    elif opt.mode == 'save_bin':
        print('start evaluate')
        main.save_data()
    else:
        raise NotImplementedError()
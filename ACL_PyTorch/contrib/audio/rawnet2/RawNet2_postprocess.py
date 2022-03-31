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
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse

sys.path.append('RawNet/python/RawNet2/')
from utils import cos_sim

def get_l_embeddings(list_embeddings,bs,path="def"):
    temp = ""
    l_embeddings = []
    index = 0
    l_utt = []
    l_code = []
    with tqdm(total=len(list_embeddings), ncols=70) as pbar:
        if bs==1:
            files = sorted(list_embeddings)
        else:
            files = list_embeddings.keys()
        for f in files:
            if bs==1:
                t = np.loadtxt(path + "/" + f)
                t = t.astype(np.float32)
            else:
                t = list_embeddings[f]
            index += 1
            key = f.replace("$", "/", 2).split("$")[0]
            if (temp == ""):
                temp = key
                l_utt.append(key)
            if (key == temp):
                l_code.append(t)
            else:
                l_embeddings.append(np.mean(l_code, axis=0))
                temp = key
                l_utt.append(key)
                l_code = []
                l_code.append(t)
            if (index == len(files)):
                l_embeddings.append(np.mean(l_code, axis=0))
            pbar.update(1)
    return l_utt,l_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='bin path', default="", required=True)
    parser.add_argument('--batch_size', help='batch size', required=True)
    parser.add_argument('--output', help='result path', default="result/")
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    base = args.input
    save_dir = args.output
    d_embeddings = {}
    if batch_size == 1:
        for path, dirs, files in os.walk(base):
            l_utt,l_embeddings = get_l_embeddings(files,batch_size,path);
            if not len(l_utt) == len(l_embeddings):
                print(len(l_utt), len(l_embeddings))
                exit()
            for k, v in zip(l_utt, l_embeddings):
                d_embeddings[k] = v
    else:
        with open('bs16_key.txt', 'r') as f:
            l_val = f.readlines()
        bs16_out = []
        for path, dirs, files in os.walk(base):
            files = sorted(files, key=lambda x: [int(x.split('_')[0])])
            for f in files:
                t = np.loadtxt(path + "/" + f)
                for i in t:
                    i.reshape(1024, )
                    bs16_out.append(i)
        bs16_out_embeddings = {}
        if not len(l_val) == len(bs16_out):
            print(len(l_val), len(bs16_out))
            exit()
        for k, v in zip(l_val, bs16_out):
            bs16_out_embeddings[k] = v
        l_utt,l_embeddings = get_l_embeddings(bs16_out_embeddings,batch_size);
        if not len(l_utt) == len(l_embeddings):
            print(len(l_utt), len(l_embeddings))
            exit()
        for k, v in zip(l_utt, l_embeddings):
            d_embeddings[k] = v

    with open('RawNet/trials/vox_original.txt', 'r') as f:
        l_val_trial = f.readlines()
    y_score = []
    y = []
    f_res = open(save_dir + 'result_detail_bs{}.txt'.format(batch_size), 'w')
    for line in l_val_trial:
        trg, utt_a, utt_b = line.strip().split(' ')
        y.append(int(trg))
        y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
        f_res.write('{score} {target}\n'.format(score=y_score[-1], target=y[-1]))
    f_res.close()
    fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    f_eer_301 = open(save_dir + 'result_eer_{}.txt'.format(batch_size), 'w')
    f_eer_301.write('bs{dir} evaluation EER: {eer}\n'.format(dir=batch_size, eer=eer))
    f_eer_301.close()
    print('bs{dir} evaluation EER: {eer}\n'.format(dir=batch_size, eer=eer))


if __name__ == '__main__':
    main()

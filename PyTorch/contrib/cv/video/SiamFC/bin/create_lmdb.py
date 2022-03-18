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

import lmdb
import cv2
import numpy as np
import os 
import hashlib
import functools
import argparse
import multiprocessing
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
multiprocessing.set_start_method('spawn', True)


def worker(video_name):
    image_names = glob(video_name+'/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv


def create_lmdb(data_dir, output_dir, num_threads):

    video_names = glob(data_dir+'/*')

    video_names = [x for x in video_names if os.path.isdir(x)]  # Annotation, Imageset, Annotation
   
    db = lmdb.open(output_dir, map_size=int(50e9))  # dir named ILSVRC_VID_CURATION.lmdb, include data.mdb and lock.mdb

    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)


Data_dir = './data/ILSVRC_VID_CURATION'
Output_dir = './data/ILSVRC_VID_CURATION.lmdb'
Num_threads = 8
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('--d', default=Data_dir, type=str, help="data_dir")
    parser.add_argument('--o', default=Output_dir, type=str, help="out put")
    parser.add_argument('--n', default=Num_threads, type=int, help="thread_num")
    args = parser.parse_args()

    create_lmdb(args.d, args.o, args.n)

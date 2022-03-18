# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import time
import os.path as osp
import torch
import glob
import re
from torchreid.metrics import distance as distance
from torchreid.metrics import rank as rank


def gen_qf(filepath):
    qf = np.zeros((3368, 512), dtype=float)
    count = 0
    for gtfile in os.listdir(filepath):  # go through all the file folder
        if osp.join(filepath, gtfile).split('_')[0] == '-1':
            continue
        else:   # is not junk img
            with open(os.path.join(filepath, gtfile), 'r') as f:  # open the file
                lines = f.readlines()
                for line in lines:
                    list = line.strip('\n').split(' ')   # string list
                    list = list[0: 512]
                i = 0
                for num in list:
                    qf[count, i] = float(num)
                    i = i + 1
                count = count + 1
    qf = torch.Tensor(qf)        
    return qf


def gen_gf(filepath):
    gf = np.zeros((15913 , 512), dtype=float)  
    count = 0
    for gtfile in os.listdir(filepath):  # go through all the file folder
        qfnum = gtfile.split('_')[0]
        if qfnum != '-1':   # is not junk img
            with open(os.path.join(filepath, gtfile), 'r') as f:  # open the file
                lines = f.readlines()
                for line in lines:
                    list = line.strip('\n').split(' ')   # string list
                    list = list[0: 512]
                list_float = []
                for num in list:
                    list_float.append(float(num))    
                gf_line = list_float   # float ist
                gf[count, :] = gf_line[0:512]
            count = count + 1 
    gf = torch.Tensor(gf)        
    return gf
    
  
def process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.txt'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue # junk images are just ignored
        assert 0 <= pid <= 1501 # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))
    return data  
 
  
def parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    return imgs, pids, camids


def _feature_extraction(data_loader):
    pids_, camids_ = [], []
    for batch_idx, data in enumerate(data_loader):
        imgs, pids, camids = parse_data_for_eval(data)
        pids_.append(pids)
        camids_.extend([camids])
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)
    return pids_, camids_


def create_visualization_statistical_result(all_cmc, mAP, result_store_path, json_file_name):
    print("Start to create json file")
    writer = open(os.path.join(result_store_path, json_file_name), 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []
    table_dict["value"].extend(
            [{"key": "R1", "value": str(all_cmc[0])},
             {"key": "mAP", "value": str(mAP)}])
    json.dump(table_dict, writer)
    writer.close()


if __name__ == '__main__':
    start = time.time()
    try:
        # txt file path
        query_target = sys.argv[1]      # result/dumpOutput_device0/
        gallery_target = sys.argv[2]    # result/dumpOutput_device1/
        result_json_path = sys.argv[3]  # ./
        json_file_name = sys.argv[4]    # result_bs1.json
    except IndexError:
        print("Stopped!")
        exit(1)
            
    # query_data gallery_data
    query_data = process_dir(dir_path=query_target, relabel = False)    
    gallery_data = process_dir(dir_path=gallery_target, relabel = False)   

    # qf gf
    qf = gen_qf(query_target)
    gf = gen_gf(gallery_target)
    
    # distmat
    distmat = distance.compute_distance_matrix(qf, gf, metric='euclidean')
    distmat = distmat.numpy()
  
    print('Extracting features from query set ...')
    q_pids, q_camids = _feature_extraction(data_loader=query_data)

    print('Extracting features from gallery set ...')
    g_pids, g_camids = _feature_extraction(data_loader=gallery_data)  
      
    all_cmc, mAP = rank.eval_market1501(distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, max_rank=50)
    print("R1")
    print(all_cmc[0])
    print("mAP")
    print(mAP)
    create_visualization_statistical_result(all_cmc=all_cmc, mAP=mAP, result_store_path=result_json_path, json_file_name=json_file_name)
    
    elapsed = (time.time() - start)
    print("Time used:", elapsed) 
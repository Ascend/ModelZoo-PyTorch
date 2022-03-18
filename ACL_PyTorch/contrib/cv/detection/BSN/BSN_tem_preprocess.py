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
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
import sys

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
if __name__ == '__main__':
    if not os.path.exists("output/BSN-TEM-preprocess/anchor_min"):
            os.makedirs("output/BSN-TEM-preprocess/anchor_min") 
    if not os.path.exists("output/BSN-TEM-preprocess/anchor_max"):
            os.makedirs("output/BSN-TEM-preprocess/anchor_max")
    if not os.path.exists("output/BSN-TEM-preprocess/feature"):
            os.makedirs("output/BSN-TEM-preprocess/feature")
    feature_path = "BSN-boundary-sensitive-network.pytorch/data/activitynet_feature_cuhk/"
    video_info_path = "BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/video_info_new.csv"
    video_anno_path = "BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json"
    temporal_scale = 100
    temporal_gap = 1. / temporal_scale
    subset = "full"
    boundary_ratio = 0.1
    anno_df = pd.read_csv(video_info_path)
    anno_database= load_json(video_anno_path)
    video_dict = {}
    for i in range(len(anno_df)):
      video_name=anno_df.video.values[i]
      video_info=anno_database[video_name]
      video_subset=anno_df.subset.values[i]
      if subset == "full":
          video_dict[video_name] = video_info
      if subset in video_subset:
          video_dict[video_name] = video_info
    video_list = list(video_dict.keys())
    print("%s subset video numbers: %d" %(subset,len(video_list)))
    
    for i in range(len(video_list)):
        video_name=video_list[i]
        anchor_xmin=[temporal_gap*i for i in range(temporal_scale)]
        anchor_xmax=[temporal_gap*i for i in range(1,temporal_scale+1)]
        video_df=pd.read_csv(feature_path+ "csv_mean_"+str(temporal_scale)+"/"+video_name+".csv")
        video_data = video_df.values[:,:]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data,0,1)
        video_data.float()
        video_data = np.array(video_data).astype(np.float32)
        video_data.tofile(os.path.join('./output/BSN-TEM-preprocess/feature/', video_name  + ".bin"))
        
        anchor_xmin = np.array(anchor_xmin)
        anchor_xmax = np.array(anchor_xmax)
        anchor_xmin.tofile(os.path.join('./output/BSN-TEM-preprocess/anchor_min/', video_name  + ".bin"))
        anchor_xmax.tofile(os.path.join('./output/BSN-TEM-preprocess/anchor_max/', video_name  + ".bin"))
        
    
    
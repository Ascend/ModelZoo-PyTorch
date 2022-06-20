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
import os
#import opts
import sys

#opt = opts.parse_opt()
#opt = vars(opt)
def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
        
if __name__ == '__main__':
    if not os.path.exists("output/BSN-PEM-preprocess/feature"):
        os.makedirs("output/BSN-PEM-preprocess/feature") 
    if not os.path.exists("output/BSN-PEM-preprocess/xmin"):
        os.makedirs("output/BSN-PEM-preprocess/xmin") 
    if not os.path.exists("output/BSN-PEM-preprocess/xmax"):
        os.makedirs("output/BSN-PEM-preprocess/xmax") 
    if not os.path.exists("output/BSN-PEM-preprocess/xmin_score"):
        os.makedirs("output/BSN-PEM-preprocess/xmin_score") 
    if not os.path.exists("output/BSN-PEM-preprocess/xmax_score"):
        os.makedirs("output/BSN-PEM-preprocess/xmax_score")
    subset = "validation"
    top_K = 1000
    video_info_path = "BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/video_info_new.csv"
    video_anno_path = "BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json"
    #video_info_path = opt["video_info"]
    #video_anno_path = opt["video_anno"]
    pgm_proposals_path = "output/PGM_proposals/"
    pgm_feature_path = "output/PGM_feature/"
    pem_feature_path ="output/BSN-PEM-preprocess/feature"
    pem_xmin_path ="output/BSN-PEM-preprocess/xmin"
    pem_xmax_path ="output/BSN-PEM-preprocess/xmax"
    pem_xmin_score_path ="output/BSN-PEM-preprocess/xmin_score"
    pem_xmax_score_path ="output/BSN-PEM-preprocess/xmax_score"
    anno_df = pd.read_csv(video_info_path)
    anno_database= load_json(video_anno_path)
    video_dict = {}
    for i in range(len(anno_df)):
        video_name=anno_df.video.values[i]
        video_info=anno_database[video_name]
        video_subset=anno_df.subset.values[i]
        #if subset == "full":
        #    video_dict[video_name] = video_info
        if subset in video_subset:
            video_dict[video_name] = video_info
    video_list = list(video_dict.keys())
    print("%s subset video numbers: %d" %(subset,len(video_list)))
    
    print("processing...")
    for i in range(len(video_list)):
        video_name = video_list[i]
        pdf=pandas.read_csv(pgm_proposals_path+video_name+".csv")
        pdf=pdf[:top_K]
        video_feature = numpy.load(pgm_feature_path + video_name+".npy")
        video_feature = video_feature[:top_K,:]
        video_feature = torch.Tensor(video_feature)
        video_xmin =pdf.xmin.values[:]
        video_xmax =pdf.xmax.values[:]
        video_xmin_score = pdf.xmin_score.values[:]
        video_xmax_score = pdf.xmax_score.values[:]
        
        #video_feature = np.array(video_feature).astype(np.float32)
        #if not [1000,32] expend to [1000.32]
        expend_num = 1000 - int(video_feature.shape[0])
        if expend_num != 0:
            video_expend = torch.zeros(expend_num,32)
            video_feature = torch.cat((video_feature,video_expend),0)       
        video_feature = np.array(video_feature).astype(np.float32)
        video_feature.tofile(os.path.join(pem_feature_path, video_name  + ".bin"))
        
        video_xmin = np.array(video_xmin)
        video_xmax = np.array(video_xmax)
        video_xmin_score = np.array(video_xmin_score)
        video_xmax_score = np.array(video_xmax_score)
        
        video_xmin.tofile(os.path.join(pem_xmin_path, video_name  + ".bin"))
        video_xmax.tofile(os.path.join(pem_xmax_path, video_name  + ".bin"))
        video_xmin_score.tofile(os.path.join(pem_xmin_score_path, video_name  + ".bin"))
        video_xmax_score.tofile(os.path.join(pem_xmax_score_path, video_name  + ".bin"))
        
        
        
        
        

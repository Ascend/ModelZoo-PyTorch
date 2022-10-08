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
import json
import torch
import argparse
import os
import numpy as np
import pandas as pd
import multiprocessing as mp

parser = argparse.ArgumentParser(description='BSN')

parser.add_argument('--result_path',default='output/PEM_results', type=str, help='Dir to save txt results')
parser.add_argument('--PEM_out_path', default='result/dumpOutput_device1', type=str, help='infer out path')
parser.add_argument('--PEM_video_xmin_path', default='output/BSN-PEM-preprocess/xmin', type=str, help='infer info path')
parser.add_argument('--PEM_video_xmax_path', default='output/BSN-PEM-preprocess/xmax', type=str, help='infer info path')
parser.add_argument('--PEM_video_xmin_score_path', default='output/BSN-PEM-preprocess/xmin_score', type=str, help='infer info path')
parser.add_argument('--PEM_video_xmax_score_path', default='output/BSN-PEM-preprocess/xmax_score', type=str, help='infer info path')
# parser.add_argument('--info_name', default='./deepspeech_om_bin.info', type=str, help='input info path')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
args = parser.parse_args()

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
    
def getDatasetDict():
    df=pd.read_csv("BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/video_info_new.csv")
    json_data= load_json("BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json")
    database=json_data
    video_dict={}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset=="validation":
            video_dict[video_name]=video_new_info
    return video_dict

def iou_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len +box_max-box_min
    #print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def Soft_NMS(df):
    df=df.sort_values(by="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    rstart=[]
    rend=[]
    rscore=[]

    while len(tscore)>0 and len(rscore)<=100:
        max_index=np.argmax(tscore)
        tmp_width = tend[max_index] -tstart[max_index]
        iou_list = iou_with_anchors(tstart[max_index],tend[max_index],tmp_width,np.array(tstart),np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list)/0.75)
        for idx in range(0,len(tscore)):
            if idx!=max_index:
                tmp_iou = iou_list[idx]
                if tmp_iou>0.65 + (0.9 - 0.65) * tmp_width:
                    tscore[idx]=tscore[idx]*iou_exp_list[idx]
            
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
                
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf

def video_post_process(video_list,video_dict):

    for video_name in video_list:
        df=pd.read_csv("./output/PEM_results/"+video_name+".csv")
    
        df['score']=df.iou_score.values[:]*df.xmin_score.values[:]*df.xmax_score.values[:]
        if len(df)>1:
            df=Soft_NMS(df)
        
        df=df.sort_values(by="score",ascending=False)
        video_info=video_dict[video_name]
        video_duration=float(video_info["duration_frame"]/16*16)/video_info["duration_frame"]*video_info["duration_second"]
        proposal_list=[]
    
        for j in range(min(100,len(df))):
            tmp_proposal={}
            tmp_proposal["score"]=df.score.values[j]
            tmp_proposal["segment"]=[max(0,df.xmin.values[j])*video_duration,min(1,df.xmax.values[j])*video_duration]
            proposal_list.append(tmp_proposal)
        result_dict[video_name[2:]]=proposal_list
        

def BSN_post_processing():
    video_dict=getDatasetDict()
    video_list=video_dict.keys()#[:100]
    video_list = list(video_list)
    global result_dict
    result_dict=mp.Manager().dict()
    
    num_videos = len(video_list)
    num_videos_per_thread = num_videos/8
    processes = []
    for tid in range(7):
        tmp_video_list = video_list[int(tid*num_videos_per_thread):int((tid+1)*num_videos_per_thread)]
        p = mp.Process(target = video_post_process,args =(tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[int(7*num_videos_per_thread):]
    p = mp.Process(target = video_post_process,args =(tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    
    result_dict = dict(result_dict)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open("./output/result_proposal.json","w")
    json.dump(output_dict,outfile)
    outfile.close()

if __name__ == '__main__':
    if not os.path.exists("output/PEM_results"):
            os.makedirs("output/PEM_results") 
    out_files = [ file  for file in os.listdir(args.PEM_out_path) if file.endswith(".bin") ]
    print("processing...")
    for i in range(len(out_files)):
        video_name = str(out_files[i])
        video_name = video_name[0:int(len(video_name)-6)]
        video_data = np.fromfile(args.PEM_out_path+'/'+out_files[i],dtype=np.float32)
        
        video_xmin = np.fromfile(args.PEM_video_xmin_path+'/'+video_name+'.bin',dtype=np.float64)
        video_xmax = np.fromfile(args.PEM_video_xmax_path+'/'+video_name+'.bin',dtype=np.float64)
        video_xmin_score = np.fromfile(args.PEM_video_xmin_score_path+'/'+video_name+'.bin',dtype=np.float64)
        video_xmax_score = np.fromfile(args.PEM_video_xmax_score_path+'/'+video_name+'.bin',dtype=np.float64)
        
        video_data = torch.tensor(video_data)
        video_xmin = torch.tensor(video_xmin)
        video_xmax = torch.tensor(video_xmax)
        video_xmin_score = torch.tensor(video_xmin_score)
        video_xmax_score = torch.tensor(video_xmax_score)
        data_num = int(video_xmin.shape[0])
        video_data = video_data[:data_num]
        
        video_data = video_data.view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()
        
        df=pd.DataFrame()
        df["xmin"]=video_xmin
        df["xmax"]=video_xmax
        df["xmin_score"]=video_xmin_score
        df["xmax_score"]=video_xmax_score
        df["iou_score"]=video_data       
        df.to_csv(args.result_path+'/'+video_name+".csv",index=False)
    print("PGM: start generating BSN_post feature")   
    BSN_post_processing()
    print("PGM: finish generate BSN_post feature")
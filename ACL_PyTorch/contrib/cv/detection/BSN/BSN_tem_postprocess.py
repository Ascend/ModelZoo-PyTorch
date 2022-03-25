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
import hydra
import argparse
import os
import numpy as np
import pandas as pd
import numpy
import pandas
import torch.multiprocessing as mp
from scipy.interpolate import interp1d


parser = argparse.ArgumentParser(description='BSN')

parser.add_argument('--result_path',default='./output/TEM_results', type=str, help='Dir to save txt results')
parser.add_argument('--TEM_out_path', default='./result/dumpOutput_device0', type=str, help='infer out path')
parser.add_argument('--TEM_anchor_xmin_path', default='./output/BSN-TEM-preprocess/anchor_min', type=str, help='infer info path')
parser.add_argument('--TEM_anchor_xmax_path', default='./output/BSN-TEM-preprocess/anchor_max', type=str, help='infer info path')

args = parser.parse_args()

columns=["action","start","end","xmin","xmax"]

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
    
def iou_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len +box_max-box_min
    jaccard = numpy.divide(inter_len, union_len)
    return jaccard

def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = numpy.maximum(anchors_min, box_min)
    int_xmax = numpy.minimum(anchors_max, box_max)
    inter_len = numpy.maximum(int_xmax - int_xmin, 0.)
    scores = numpy.divide(inter_len, len_anchors)
    return scores

def generateProposals(video_list,video_dict):
    tscale = 100   
    tgap = 1./tscale
    peak_thres= 0.5

    for video_name in video_list:
        tdf=pandas.read_csv("./output/TEM_results/"+video_name+".csv")
        start_scores=tdf.start.values[:]
        end_scores=tdf.end.values[:]
        
        max_start = max(start_scores)
        max_end = max(end_scores)
        
        start_bins=numpy.zeros(len(start_scores))
        start_bins[[0,-1]]=1
        for idx in range(1,tscale-1):
            if start_scores[idx]>start_scores[idx+1] and start_scores[idx]>start_scores[idx-1]:
                start_bins[idx]=1
            elif start_scores[idx]>(peak_thres*max_start):
                start_bins[idx]=1
                    
        end_bins=numpy.zeros(len(end_scores))
        end_bins[[0,-1]]=1
        for idx in range(1,tscale-1):
            if end_scores[idx]>end_scores[idx+1] and end_scores[idx]>end_scores[idx-1]:
                end_bins[idx]=1
            elif end_scores[idx]>(peak_thres*max_end):
                end_bins[idx]=1
        
        xmin_list=[]
        xmin_score_list=[]
        xmax_list=[]
        xmax_score_list=[]
        for j in range(tscale):
            if start_bins[j]==1:
                xmin_list.append(tgap/2+tgap*j)
                xmin_score_list.append(start_scores[j])
            if end_bins[j]==1:
                xmax_list.append(tgap/2+tgap*j)
                xmax_score_list.append(end_scores[j])
                
        new_props=[]
        for ii in range(len(xmax_list)):
            tmp_xmax=xmax_list[ii]
            tmp_xmax_score=xmax_score_list[ii]
            
            for ij in range(len(xmin_list)):
                tmp_xmin=xmin_list[ij]
                tmp_xmin_score=xmin_score_list[ij]
                if tmp_xmin>=tmp_xmax:
                    break
                new_props.append([tmp_xmin,tmp_xmax,tmp_xmin_score,tmp_xmax_score])
        new_props=numpy.stack(new_props)
        
        col_name=["xmin","xmax","xmin_score","xmax_score"]
        new_df=pandas.DataFrame(new_props,columns=col_name)  
        new_df["score"]=new_df.xmin_score*new_df.xmax_score
        
        new_df=new_df.sort_values(by="score",ascending=False)
        
        video_info=video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        
        try:
            gt_xmins=[]
            gt_xmaxs=[]
            for idx in range(len(video_info["annotations"])):
                gt_xmins.append(video_info["annotations"][idx]["segment"][0]/corrected_second)
                gt_xmaxs.append(video_info["annotations"][idx]["segment"][1]/corrected_second)
            new_iou_list=[]
            for j in range(len(new_df)):
                tmp_new_iou=max(iou_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
                new_iou_list.append(tmp_new_iou)
                
            new_ioa_list=[]
            for j in range(len(new_df)):
                tmp_new_ioa=max(ioa_with_anchors(new_df.xmin.values[j],new_df.xmax.values[j],gt_xmins,gt_xmaxs))
                new_ioa_list.append(tmp_new_ioa)
            new_df["match_iou"]=new_iou_list
            new_df["match_ioa"]=new_ioa_list
        except:
            pass
        new_df.to_csv("./output/PGM_proposals/"+video_name+".csv",index=False)


def getDatasetDict():
    df=pandas.read_csv("./BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/video_info_new.csv")
    json_data= load_json("./BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json")
    database=json_data
    video_dict = {}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_new_info['annotations']=video_info['annotations']
        video_new_info['subset'] = df.subset.values[i]
        video_dict[video_name]=video_new_info
    return video_dict

def generateFeature(video_list,video_dict):

    num_sample_start=8
    num_sample_end=8
    num_sample_action=16
    num_sample_interpld = 3

    for video_name in video_list:
        adf=pandas.read_csv("./output/TEM_results/"+video_name+".csv")
        score_action=adf.action.values[:]
        seg_xmins = adf.xmin.values[:]
        seg_xmaxs = adf.xmax.values[:]
        video_scale = len(adf)
        video_gap = seg_xmaxs[0] - seg_xmins[0]
        video_extend = video_scale / 4 + 10
        pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv")
        video_subset = video_dict[video_name]['subset']
        if video_subset == "training":
            pdf=pdf[:500]
        else:
            pdf=pdf[:1000]
        tmp_zeros=numpy.zeros([int(video_extend)])    
        score_action=numpy.concatenate((tmp_zeros,score_action,tmp_zeros))
        tmp_cell = video_gap
        #print('video_extend:{}'.format(video_extend))
        tmp_x = [-tmp_cell/2-(video_extend-1-ii)*tmp_cell for ii in range(int(video_extend))] + \
                 [tmp_cell/2+ii*tmp_cell for ii in range(int(video_scale))] + \
                  [tmp_cell/2+seg_xmaxs[-1] +ii*tmp_cell for ii in range(int(video_extend))]
        f_action=interp1d(tmp_x,score_action,axis=0)
        feature_bsp=[]
    
        for idx in range(len(pdf)):
            xmin=pdf.xmin.values[idx]
            xmax=pdf.xmax.values[idx]
            xlen=xmax-xmin
            xmin_0=xmin-xlen * 0.2
            xmin_1=xmin+xlen * 0.2
            xmax_0=xmax-xlen * 0.2
            xmax_1=xmax+xlen * 0.2
            #start
            plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
            plen_sample = plen_start / num_sample_interpld
            tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
            tmp_y_new_start_action=f_action(tmp_x_new)
            tmp_y_new_start = [numpy.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
            #end
            plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
            plen_sample = plen_end / num_sample_interpld
            tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 
            tmp_y_new_end_action=f_action(tmp_x_new)
            tmp_y_new_end = [numpy.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
            #action
            plen_action= (xmax-xmin)/(num_sample_action-1)
            plen_sample = plen_action / num_sample_interpld
            tmp_x_new = [ xmin - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 
            tmp_y_new_action=f_action(tmp_x_new)
            tmp_y_new_action = [numpy.mean(tmp_y_new_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]
            tmp_feature = numpy.concatenate([tmp_y_new_action,tmp_y_new_start,tmp_y_new_end])
            feature_bsp.append(tmp_feature)
        feature_bsp = numpy.array(feature_bsp)
        numpy.save("./output/PGM_feature/"+video_name,feature_bsp)



def PGM_proposal_generation():
    video_dict= load_json("./BSN-boundary-sensitive-network.pytorch/data/activitynet_annotations/anet_anno_action.json")
    video_list=video_dict.keys()#[:199]
    video_list = list(video_list)
    num_videos = len(video_list)
    num_videos_per_thread = num_videos/8
    processes = []
    for tid in range(7):
        tmp_video_list = video_list[int(tid*num_videos_per_thread):int((tid+1)*num_videos_per_thread)]
        p = mp.Process(target = generateProposals,args =(tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    
    tmp_video_list = video_list[int(7*num_videos_per_thread):]
    p = mp.Process(target = generateProposals,args =(tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()
def PGM_feature_generation():
    video_dict=getDatasetDict()
    video_list=video_dict.keys()
    video_list = list(video_list)
    num_videos = len(video_list)
    num_videos_per_thread = num_videos/8
    processes = []
    for tid in range(7):
        tmp_video_list = video_list[int(tid*num_videos_per_thread):int((tid+1)*num_videos_per_thread)]
        p = mp.Process(target = generateFeature,args =(tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    
    tmp_video_list = video_list[int(7*num_videos_per_thread):]
    p = mp.Process(target = generateFeature,args =(tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == '__main__':
    out_files = os.listdir(args.TEM_out_path)
    if not os.path.exists("output/TEM_results"):
            os.makedirs("output/TEM_results") 
    print("processing...")
    for i in range(len(out_files)):
        video_name = str(out_files[i])
        video_name = video_name[0:int(len(video_name)-6)]
        video_data = np.fromfile(args.TEM_out_path+'/'+out_files[i],dtype=np.float32)
        #print(video_data)
        video_data = torch.tensor(video_data.reshape(1,3,100))
        #video_data.reshape(1,3,1000)
        video_data = video_data.detach().cpu().numpy()
        
        anchor_xmin = np.fromfile(args.TEM_anchor_xmin_path+'/'+video_name+'.bin',dtype=np.float64)
        anchor_xmax = np.fromfile(args.TEM_anchor_xmax_path+'/'+video_name+'.bin',dtype=np.float64)
        
        anchor_xmin = torch.tensor(anchor_xmin)
        anchor_xmax = torch.tensor(anchor_xmax)
        video_action = video_data[:,0,:]
        video_start = video_data[:,1,:]
        video_end = video_data[:,2,:]
    
        video_result = np.stack((video_action[0],video_start[0],video_end[0],anchor_xmin,anchor_xmax),axis=1)
        
        video_df = pd.DataFrame(video_result,columns=columns) 
        video_df.to_csv(args.result_path+"/"+video_name+".csv",index=False) 
      
    if not os.path.exists("output/PGM_proposals"):
        os.makedirs("output/PGM_proposals") 
    print("PGM: start generating proposals")
    PGM_proposal_generation()
    print("PGM: finish generate proposals")
        
    if not os.path.exists("output/PGM_feature"):
        os.makedirs("output/PGM_feature") 
    print("PGM: start generating BSP feature")
    PGM_feature_generation()
    print("PGM: finish generate BSP feature")
        
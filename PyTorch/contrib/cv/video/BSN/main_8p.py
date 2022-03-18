# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
NPU_WORLD_SIZE = int(os.getenv('NPU_WORLD_SIZE'))
RANK = int(os.getenv('RANK'))
torch.distributed.init_process_group('hccl', rank=RANK, world_size=NPU_WORLD_SIZE)
sys.dont_write_bytecode = True
import os
import json
import time
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from dataset import VideoDataSet,ProposalDataSet
from models import TEM,PEM
from loss_function import TEM_loss_function,PEM_loss_function
import apex
from apex import amp
import pandas as pd
from pgm import PGM_proposal_generation,PGM_feature_generation
from post_processing import BSN_post_processing
from eval import evaluation_proposal

## 0917: del all writer in tensorboardX
def train_TEM(data_loader,model,optimizer,epoch,opt):
    model.train()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        # measure data loading time
        #start_time = time.time()
        if n_iter == 5:
            time_step5 = time.time()
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        cost = loss["cost"] 
        optimizer.zero_grad()
        #cost.backward()
        with amp.scale_loss(cost, optimizer) as scaled_loss:
           scaled_loss.backward()
        optimizer.step()
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()
        # fps
        if NPU_CALCULATE_DEVICE == 0:
            if (n_iter + 1) == len(data_loader):
                     time_avg = time.time() - time_step5
                     fps =  NPU_WORLD_SIZE * opt["tem_batch_size"] * len(data_loader) / time_avg
                     print("[epoch %d][%d/%d] FPS(TEM): %.4f time_avg: %.4f" %
                           (epoch + 1, n_iter + 1, len(data_loader), fps, time_avg))
        #print("Iter:{} || MaxIter:{} || Time:{:.4f} || Epoch:{} || MaxEpoch:{}".format(n_iter, len(data_loader), time.time()-start_time, epoch, opt["tem_epoch"]))
    print("TEM training loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch + 1,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1)))

def test_TEM(data_loader,model,epoch,opt):
    model.eval()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter,(input_data,label_action,label_start,label_end) in enumerate(data_loader):
        TEM_output = model(input_data)
        loss = TEM_loss_function(label_action,label_start,label_end,TEM_output,opt)
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()  
    print("TEM testing  loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" %(epoch + 1,epoch_action_loss/(n_iter+1),
                                                                                        epoch_start_loss/(n_iter+1),
                                                                                        epoch_end_loss/(n_iter+1)))
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/tem_checkpoint.pth.tar" )
    if epoch_cost< model.module.tem_best_loss:
        model.module.tem_best_loss = np.mean(epoch_cost)
        torch.save(state, opt["checkpoint_path"]+"/tem_best.pth.tar" )

def train_PEM(data_loader,model,optimizer,epoch,opt):
    model.train()
    epoch_iou_loss = 0 
    for n_iter,(input_data,label_iou) in enumerate(data_loader):
        #start_time = time.time()
        if n_iter == 5:
            time_step5 = time.time()
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,label_iou,model,opt)
        optimizer.zero_grad()
        #iou_loss.backward()
        with amp.scale_loss(iou_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        epoch_iou_loss += iou_loss.cpu().detach().numpy()
        if NPU_CALCULATE_DEVICE == 0:
            if (n_iter + 1) == len(data_loader):
                time_avg = time.time() - time_step5
                fps =  NPU_WORLD_SIZE * opt["pem_batch_size"] * len(data_loader) / time_avg
                print("[epoch %d][%d/%d] FPS(PEM): %.4f time_avg: %.4f" %
                           (epoch + 1, n_iter + 1, len(data_loader), fps, time_avg))
        #print("Iter:{} || MaxIter:{} || Time:{:.4f} || Epoch:{} || MaxEpoch:{}".format(n_iter,
        #    len(data_loader), time.time()-start_time, epoch, opt["pem_epoch"]))  
    print("PEM training loss(epoch %d): iou - %.04f" %(epoch + 1,epoch_iou_loss/(n_iter+1)))

def test_PEM(data_loader,model,epoch,opt):
    model.eval()
    epoch_iou_loss = 0
    for n_iter,(input_data,label_iou) in enumerate(data_loader):
        PEM_output = model(input_data)
        iou_loss = PEM_loss_function(PEM_output,label_iou,model,opt)
        epoch_iou_loss += iou_loss.cpu().detach().numpy()  
    print("PEM testing  loss(epoch %d): iou - %.04f" %(epoch + 1,epoch_iou_loss/(n_iter+1)))  
    state = {'epoch': epoch + 1,
                'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"]+"/pem_checkpoint.pth.tar" )
    if epoch_iou_loss<model.module.pem_best_loss :
        model.module.pem_best_loss = np.mean(epoch_iou_loss)
        torch.save(state, opt["checkpoint_path"]+"/pem_best.pth.tar" )


def BSN_Train_TEM(opt):
    model = TEM(opt)
    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    optimizer = optim.Adam(model.parameters(),lr=opt["tem_training_lr"],weight_decay = opt["tem_weight_decay"])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128)
    # 1216: add NPUFusedSGD
    #optimizer = apex.optimizers.NpuFusedSGD(model.parameters(),lr=opt["tem_training_lr"],weight_decay = opt["tem_weight_decay"])
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128, combine_grad=True)
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)
    #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()  
    train_loader_sampler = torch.utils.data.distributed.DistributedSampler(VideoDataSet(opt,subset="train"))
    train_loader_batch_size = int(model.module.batch_size / int(os.getenv('NPU_WORLD_SIZE')))
    # 0917: numberworker 8 -> 4
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="train"),
                                                batch_size=train_loader_batch_size, shuffle=False,
                                                num_workers=4, pin_memory=False,drop_last=True, sampler = train_loader_sampler)            
    test_loader_sampler = torch.utils.data.distributed.DistributedSampler(VideoDataSet(opt,subset="validation"))
    test_loader_batch_size = int(model.module.batch_size / int(os.getenv('NPU_WORLD_SIZE')))
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="validation"),
                                                batch_size=test_loader_batch_size, shuffle=False,
                                                num_workers=4, pin_memory=False,drop_last=True, sampler = test_loader_sampler)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["tem_step_size"], gamma = opt["tem_step_gamma"])      
    for epoch in range(opt["tem_epoch"]):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        scheduler.step()
        train_TEM(train_loader,model,optimizer,epoch,opt)
        test_TEM(test_loader,model,epoch,opt)  


def BSN_Train_PEM(opt):
    model = PEM(opt)
    model = model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    optimizer = optim.Adam(model.parameters(),lr=opt["pem_training_lr"],weight_decay = opt["pem_weight_decay"])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128)
    # 1216: add NPUFusedSGD
    #optimizer = apex.optimizers.NpuFusedSGD(model.parameters(),lr=opt["pem_training_lr"],weight_decay = opt["pem_weight_decay"])
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128, combine_grad=True)
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)
    #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    def collate_fn(batch):
        batch_data = torch.cat([x[0] for x in batch])
        batch_iou = torch.cat([x[1] for x in batch])
        return batch_data,batch_iou
    train_loader_sampler = torch.utils.data.distributed.DistributedSampler(ProposalDataSet(opt,subset="train"))
    train_loader_batch_size = int(model.module.batch_size / int(os.getenv('NPU_WORLD_SIZE'))) 
    train_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="train"),
                                                batch_size=train_loader_batch_size, shuffle=False,
                                                num_workers=8, pin_memory=False,drop_last=True,collate_fn=collate_fn, sampler = train_loader_sampler)            
    test_loader_sampler = torch.utils.data.distributed.DistributedSampler(ProposalDataSet(opt,subset="validation"))
    test_loader_batch_size = int(model.module.batch_size / int(os.getenv('NPU_WORLD_SIZE')))
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset="validation"),
                                                batch_size=test_loader_batch_size, shuffle=False,
                                                num_workers=8, pin_memory=False,drop_last=True,collate_fn=collate_fn, sampler = test_loader_sampler)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["pem_step_size"], gamma = opt["pem_step_gamma"]) 
    for epoch in range(opt["pem_epoch"]):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        scheduler.step()
        train_PEM(train_loader,model,optimizer,epoch,opt)
        test_PEM(test_loader,model,epoch,opt)

def BSN_inference_TEM(opt):
    model = TEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/tem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).npu()
    model.eval()
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="full"),
                                                batch_size=64, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
    columns=["action","start","end","xmin","xmax"]
    for index_list,input_data,anchor_xmin,anchor_xmax in test_loader:
        TEM_output = model(input_data).detach().cpu().numpy()
        batch_action = TEM_output[:,0,:]
        batch_start = TEM_output[:,1,:]
        batch_end = TEM_output[:,2,:]
        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax]) 
        for batch_idx,full_idx in enumerate(index_list):            
            print("full_idx=================",full_idx)
            video = test_loader.dataset.video_list[full_idx]
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]    
            video_result = np.stack((video_action,video_start,video_end,anchor_xmin,anchor_xmax),axis=1)
            video_df = pd.DataFrame(video_result,columns=columns)  
            video_df.to_csv("./output/TEM_results/"+video+".csv",index=False)    
            
def BSN_inference_PEM(opt):
    model = PEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"]+"/pem_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).npu()
    model.eval()
    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt,subset=opt["pem_inference_subset"]),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True,drop_last=False)
    for idx,(video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score) in enumerate(test_loader):
        print("idx===================", idx)
        video_name = test_loader.dataset.video_list[idx]
        video_conf = model(video_feature).view(-1).detach().cpu().numpy()
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()
        df=pd.DataFrame()
        df["xmin"]=video_xmin
        df["xmax"]=video_xmax
        df["xmin_score"]=video_xmin_score
        df["xmax_score"]=video_xmax_score
        df["iou_score"]=video_conf
        df.to_csv("./output/PEM_results/"+video_name+".csv",index=False)


def main(opt):
    if opt["module"] == "TEM":
        if opt["mode"] == "train":
            print("TEM training start")  
            BSN_Train_TEM(opt)
            print("TEM training finished")  
        elif opt["mode"] == "inference": 
            if not os.path.exists("output/TEM_results"):
                os.makedirs("output/TEM_results") 
            print ("TEM inference start")
            BSN_inference_TEM(opt)
            print ("TEM inference finished")
            if not os.path.exists("output/PGM_proposals"):
                os.makedirs("output/PGM_proposals") 
            print ("PGM: start generate proposals")
            PGM_proposal_generation(opt)
            print ("PGM: finish generate proposals") 
            if not os.path.exists("output/PGM_feature"):
                os.makedirs("output/PGM_feature") 
            print("PGM: start generate BSP feature")
            PGM_feature_generation(opt)
            print("PGM: finish generate BSP feature")
    elif opt["module"] == "PEM":
        if opt["mode"] == "train":
            print("PEM training start")  
            BSN_Train_PEM(opt)
            print("PEM training finished")  
        elif opt["mode"] == "inference":
            if not os.path.exists("output/PEM_results"):
                os.makedirs("output/PEM_results") 
            print("PEM inference start")  
            BSN_inference_PEM(opt)
            print("PEM inference finished")        
            print("Post processing start")
            BSN_post_processing(opt)
            print("Post processing finished")
            evaluation_proposal(opt)            
    print("")
        
if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    main(opt)

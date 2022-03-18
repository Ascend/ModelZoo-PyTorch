# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import sys
from dataset import VideoDataSet
from loss_function import bmn_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal
from apex import amp

sys.dont_write_bytecode = True


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.npu()
        label_start = label_start.npu()
        label_end = label_end.npu()
        label_confidence = label_confidence.npu()
        #print('1')
       
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.npu())
        #print('2')
        optimizer.zero_grad()
        with amp.scale_loss(loss[0].npu(), optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss[0].npu().backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        print(
            "BMN training loss(epoch %d, n_iter %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
                epoch, n_iter, epoch_tem_loss / (n_iter + 1),
                epoch_pemclr_loss / (n_iter + 1),
                epoch_pemreg_loss / (n_iter + 1),
                epoch_loss / (n_iter + 1)))
        #if n_iter>200:
          #break
        
        
def test_BMN(data_loader, model, epoch, bm_mask):
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.npu()
        label_start = label_start.npu()
        label_end = label_end.npu()
        label_confidence = label_confidence.npu()
        #print('1')
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.npu())
        #print('2')
        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        print(
            "BMN test loss(epoch %d, n_iter %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
                epoch, n_iter, epoch_tem_loss / (n_iter + 1),
                epoch_pemclr_loss / (n_iter + 1),
                epoch_pemreg_loss / (n_iter + 1),
                epoch_loss / (n_iter + 1)))
        #if n_iter>200:
          #break
    print(torch.npu.synchronize(),'test finish')
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")
    print(torch.npu.synchronize(),'save finish')
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")


def BMN_Train(opt):
    model = BMN(opt)
    model = model.npu()
    #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=opt["loss_scale"])
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=False, drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=True,
                                              num_workers=8, pin_memory=False, drop_last=True)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["train_epochs"]):
        print(torch.npu.synchronize(),"train_BMN begin")
        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        print(torch.npu.synchronize(),"train_BMN success")
        
        print(torch.npu.synchronize(),"test_BMN begin")
        test_BMN(test_loader, model, epoch, bm_mask)
        print(torch.npu.synchronize(),"test_BMN success")
        
        scheduler.step()


def BMN_inference(opt):
    model = BMN(opt)
    model = model.npu()
    #model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_best.pth.tar", map_location='npu:0')
    #base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    #model.load_state_dict(base_dict)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.npu()
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    
    torch.npu.set_device(0)
    if opt["mode"] == "train":
        BMN_Train(opt)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference(opt)
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)
    elif opt["mode"] == "together":
        BMN_Train(opt)        
        
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference(opt)
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)
     
if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)

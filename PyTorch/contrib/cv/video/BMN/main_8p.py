# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Huawei Technologies Co., Ltd
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
import torch
if torch.__version__ >= "1.8":
    import torch_npu
print(torch.__version__)
import sys
from dataset import VideoDataSet
import os
import json
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal
from apex import amp
import torch.distributed as dist
import time
import torch.npu
from loss_function import bmn_loss_func, get_mask

sys.dont_write_bytecode = True
    
def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0

    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        if opt["local_rank"] == 0:
            if n_iter == 0:
                time_iter0 = time.time()
            if n_iter == 5:
                time_iter5 = time.time()
        input_data = input_data.npu()
        label_start = label_start.npu()
        label_end = label_end.npu()
        label_confidence = label_confidence.npu()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.npu())
        if opt["world_size"] == 1 and epoch == 0 and n_iter == 6:
            with torch.autograd.profiler.profile(use_npu=True) as prof: 
                optimizer.zero_grad()
                with amp.scale_loss(loss[0], optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss[0].backward()
                optimizer.step()
            #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace("910A_1p.prof") # "output.prof"为输出文件地址
        else:
            optimizer.zero_grad()
            with amp.scale_loss(loss[0], optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss[0].backward()
            optimizer.step()
        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        
        epoch_loss += loss[0].cpu().detach().numpy()
        
        if opt["local_rank"] == 0:
            time_avg = time.time() - time_iter0
            #print("n_iter %d, time: %.2f"%(n_iter, time_avg)) 
        
    if (n_iter + 1) == (data_loader.dataset.__len__() // (opt["batch_size"])) and opt["local_rank"] == 0:
        time_avg = time.time() - time_iter5
        fps = (opt["batch_size"]) * (data_loader.dataset.__len__() // (opt["batch_size"])) / time_avg
        #sum_fps += fps  
        #avg_fps = sum_fps/(float(epoch + 1))
        print("Epoch: %d,FPS: %.2f,time: %.2f"%(epoch, fps, time_avg))  
                        
    if opt["local_rank"] == 0:
        print(
            "BMN training loss(epoch %d, n_iter %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, n_iter, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

def test_BMN(data_loader, model, epoch, bm_mask):
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        if n_iter == 0 and opt["local_rank"] == 0:
            time_iter0 = time.time()
        input_data = input_data.npu()
        label_start = label_start.npu()
        label_end = label_end.npu()
        label_confidence = label_confidence.npu()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.npu())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        
        if opt["local_rank"] == 0:
            time_avg = time.time() - time_iter0
            #print("n_iter %d, time: %.2f"%(n_iter, time_avg))
            
    if opt["local_rank"] == 0:
        print(
            "BMN test loss(epoch %d, n_iter %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
                epoch, n_iter, epoch_tem_loss / (n_iter + 1),
                epoch_pemclr_loss / (n_iter + 1),
                epoch_pemreg_loss / (n_iter + 1),
                epoch_loss / (n_iter + 1)))
  
    if opt["local_rank"] == 0:
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")


def BMN_Train(opt):
    model = BMN(opt)
    #model = model.npu()
    model = model.to(f'npu:{opt["local_rank"]}')
    print(model)

    if opt["finetune"] == 1:
        checkpoint = torch.load(opt["pth_path"], map_location='npu:0')
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model.load_state_dict(base_dict)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],weight_decay=opt["weight_decay"])
    
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=128, combine_grad=True)
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt["local_rank"]], broadcast_buffers=False)
    
    train_loader_sampler = torch.utils.data.distributed.DistributedSampler(VideoDataSet(opt, subset="train"))
    train_loader_batch_size = int(opt["batch_size"] / int(opt["world_size"]))
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=train_loader_batch_size, shuffle=False,
                                               num_workers=8, pin_memory=False, drop_last = True, sampler = train_loader_sampler)
    test_loader_sampler = torch.utils.data.distributed.DistributedSampler(VideoDataSet(opt, subset="validation"))
    test_loader_batch_size = int(opt["batch_size"] / int(opt["world_size"]))

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=test_loader_batch_size, shuffle=False,
                                              num_workers=8, pin_memory=False, drop_last = True, sampler = test_loader_sampler)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    
    for epoch in range(opt["train_epochs"]):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        test_BMN(test_loader, model, epoch, bm_mask)    
        scheduler.step()


def BMN_inference(opt):
    model = BMN(opt)
    model = model.to('npu:0')

    checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_best.pth.tar", map_location='npu:0')
    #model.load_state_dict(checkpoint['state_dict'])
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
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
    
    if opt["is_distributed"] == 0:
        torch.npu.set_device(0)
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29681'
        #print('world_size: ', opt["world_size"])
        #print('rank: ', opt["local_rank"])
        dist.init_process_group(backend='hccl',world_size=opt["world_size"], rank=opt["local_rank"])
        local_device = torch.device(f'npu:{opt["local_rank"]}')
        torch.npu.set_device(local_device)
        if opt["local_rank"] == 0:
            print("using npu :{}".format(opt["DeviceID"]))
        # declare instance for GAN

    opt["feature_path"] = opt["data_path"]
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
    elif opt["mode"] == "full":
        opt["mode"] = "train"
        BMN_Train(opt)     
        if opt["local_rank"] == 0:
            opt["mode"] = "inference"
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
    print(opt)
    main(opt)

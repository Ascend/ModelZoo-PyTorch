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
import sys
from dataset import VideoDataSet
import os
import json
import torch
import torch.nn.parallel
import numpy as np
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal
import torch.npu


sys.dont_write_bytecode = True
    
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
                                              num_workers=1, pin_memory=True, drop_last=False)
                                              
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(test_loader):
            if n_iter == 0:
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
                new_df.to_csv("./output/demo/demo.csv", index=False)
                print("demo result stored in:/output/demo/")
                break

def main(opt):
      
      local_device = torch.device("npu:0")
      torch.npu.set_device(local_device)
              
      opt["mode"] == "inference"
      if not os.path.exists("output/demo"):
          os.makedirs("output/demo")
      BMN_inference(opt)

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

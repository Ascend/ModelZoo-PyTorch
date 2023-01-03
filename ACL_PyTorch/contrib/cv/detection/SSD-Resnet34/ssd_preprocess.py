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
import os
import torch
import torch.utils.data.distributed
import torch.distributed as dist

from parse_config import parse_args, validate_arguments, validate_group_bn
from data.build_pipeline import build_pipeline
from data.prefetcher import eval_prefetcher
from eval import setup_distributed
import tqdm

def preprocess(args,coco):
    coco = eval_prefetcher(iter(coco),
                           torch.device('cpu'),
                           args.pad_input,
                           args.nhwc,
                           args.use_fp16)
    for nbatch, (img, img_id, img_size) in tqdm.tqdm(enumerate(coco)):
        with torch.no_grad():
            # print("img_id=",img_id)
            bin_name=str(img_id)+ ".bin"
            bin_fl = bin_output +'/'+ bin_name
            img=img.detach().cpu()
            img = img.numpy()
            img.tofile(bin_fl)
    return

def run(args):
    args = setup_distributed(args)
    val_loader, inv_map,cocoGt = build_pipeline(args, training=False)
    preprocess(args,val_loader)

if __name__ == "__main__":
    args = parse_args()
    validate_arguments(args)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    
    bin_output=args.bin_output
    
    if not os.path.exists(bin_output):
        os.makedirs(bin_output)

    run(args)

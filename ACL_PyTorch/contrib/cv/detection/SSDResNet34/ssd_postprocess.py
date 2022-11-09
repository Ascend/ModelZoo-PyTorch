# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch
import time
import numpy as np
import torch.utils.data.distributed

from parse_config import parse_args, validate_arguments
from data.build_pipeline import build_pipeline
from box_coder import build_ssd300_coder
from async_evaluator import AsyncEvaluator
from eval import print_message, evaluate_coco, setup_distributed
from data.prefetcher import eval_prefetcher

def coco_eval(args, coco, cocoGt, encoder, inv_map, epoch, evaluator=None):        
    bin_input=args.bin_input
    
    threshold = args.threshold
    local_rank = args.local_rank

    ret = []
    overlap_threshold = 0.50
    nms_max_detections = 200
    start = time.time()
    
    coco = eval_prefetcher(iter(coco),
                           torch.device('cpu'),
                           args.pad_input,
                           args.nhwc,
                           args.use_fp16)
    
    for nbatch, (img, img_id, img_size) in enumerate(coco):
        with torch.no_grad():
            ploc = np.fromfile(bin_input + '/' + str(img_id) + '_0.bin', np.float32)
            ploc = np.reshape(ploc, (1, 4, 8732))
            ploc=torch.from_numpy(ploc).cpu()

            plabel = np.fromfile(bin_input + '/' + str(img_id) + '_1.bin', np.float32)
            plabel = np.reshape(plabel, (1, 81, 8732))
            plabel = torch.from_numpy(plabel).cpu()

            ploc, plabel = ploc.float(), plabel.float()
            
            print("img_id=", img_id)

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                
                result = encoder.decode_batch(ploc_i, plabel_i, overlap_threshold, nms_max_detections)

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r[0].cpu().numpy() for r in result]

                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0]*wtot, \
                                        loc_[1]*htot,
                                        (loc_[2] - loc_[0])*wtot,
                                        (loc_[3] - loc_[1])*htot,
                                        prob_,
                                        inv_map[(label_+1)]])
                
    ret = np.array(ret).astype(np.float32)
    final_results = ret
    print_message(args.rank, "Predicting Ended, total time: {:.2f} s".format(time.time()-start))
    evaluator.submit_task(epoch, evaluate_coco, final_results, cocoGt, local_rank, threshold)
    return
    
def postprocess(args):
    args = setup_distributed(args)
    encoder = build_ssd300_coder()
    evaluator = AsyncEvaluator(num_threads=1)
    val_loader, inv_map, cocoGt = build_pipeline(args, training=False)
    coco_eval(args,
              val_loader,
              cocoGt,
              encoder,
              inv_map,
              0,
              evaluator=evaluator)
    return False

def main():
    args = parse_args()
    args.evaluation.sort()
    validate_arguments(args)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = not args.profile_cudnn_get
    success = postprocess(args)

if __name__ == "__main__":
    main()
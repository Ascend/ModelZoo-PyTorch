# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
if torch.__version__ >= '1.8':
    import torch_npu
import time
import numpy as np
import io

from ssd300 import SSD300
from box_coder import dboxes300_coco, build_ssd300_coder
from parse_config import parse_args, validate_arguments, validate_group_bn
from data.build_pipeline import build_pipeline
from data.prefetcher import eval_prefetcher
from async_evaluator import AsyncEvaluator

import sys

# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist

# Apex imports
try:
    import apex_C
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier
    #import amp_C
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")

# from SSD import _C as C

def print_message(rank, *print_args):
    if rank == 0:
        print(*print_args)

"""
Take results and produce mAP on COCO

Intended to be used with an async evaluator, and run on a single
node -- calling code is responsible for that delegation
"""
def evaluate_coco(final_results, cocoGt, local_rank, threshold):
    from pycocotools.cocoeval import COCOeval
    #cocoDt = cocoGt.loadRes(final_results, use_ext=True)
    cocoDt = cocoGt.loadRes(final_results)
    #E = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=True)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))
    sys.stdout.flush()

    return E.stats[0]

def coco_eval(args, model, coco, cocoGt, encoder, inv_map, epoch, iteration, evaluator=None):
    from pycocotools.cocoeval import COCOeval

    threshold = args.threshold
    batch_size = args.eval_batch_size
    use_fp16 = args.use_fp16
    local_rank = args.local_rank
    N_gpu = args.N_gpu
    use_nhwc = args.nhwc
    pad_input = args.pad_input
    distributed = args.distributed

    ret = []
    overlap_threshold = 0.50
    nms_max_detections = 200
    start = time.time()

    # Wrap dataloader for prefetching
    coco = eval_prefetcher(iter(coco),
                           torch.npu.current_device(),
                           args.pad_input,
                           args.nhwc,
                           args.use_fp16)
    for nbatch, (img, img_id, img_size) in enumerate(coco):
        with torch.no_grad():
            # Get predictions

            ploc, plabel = model(img)
            ploc = torch_npu.npu_format_cast(ploc, 2)
            plabel = torch_npu.npu_format_cast(plabel, 2)
            ploc, plabel = ploc.float(), plabel.float()
            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
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

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).npu()

        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).npu() for _ in range(N_gpu)]
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).npu())

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size-ret_copy.shape[0], 7, dtype=torch.float32).npu()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).npu() for i in range(N_gpu)]
        # Everyone exchanges (padded) results
        torch.distributed.all_gather(other_ret, ret_pad)

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
        # Otherwise full results are just our results
        final_results = ret

    print_message(args.rank, "Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    # All results are assembled -- if rank == 0 start async evaluation (if enabled)
    if args.rank == 0 and (evaluator is not None):
        evaluator.submit_task(epoch, evaluate_coco, final_results, cocoGt, local_rank, threshold)

    return


def load_checkpoint(model, checkpoint):
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'module' from checkpoint
    saved_model = od["model"]
    for k in list(saved_model.keys()):
        if k.startswith('module.'):
            saved_model[k[7:]] = saved_model.pop(k)
    model.load_state_dict(saved_model)

def setup_distributed(args):
    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    if args.distributed:
        torch.npu.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='hccl',
                                            world_size=int(os.environ['WORLD_SIZE']),
                                            rank=args.local_rank,
                                             )
    args.local_seed = 0
    # start timing here
    if args.distributed:
        args.N_gpu = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        args.N_gpu = 1
        args.rank = 0

    validate_group_bn(args.bn_group)

    return args

# setup everything (model, etc) to run eval
def run_eval(args):
    args = setup_distributed(args)

    from pycocotools.coco import COCO

    local_seed = args.local_seed
    encoder = build_ssd300_coder()

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")


    val_loader, inv_map, cocoGt = build_pipeline(args, training=False)

    model_options = {
        'use_nhwc' : args.nhwc,
        'pad_input' : args.pad_input,
        'bn_group' : args.bn_group,
        'pretrained' : False,
    }

    ssd300_eval = SSD300(args, args.num_classes, **model_options).npu()
    if args.use_fp16:
        convert_network(ssd300_eval, torch.half)
    ssd300_eval.eval()

    if args.checkpoint is not None:
        load_checkpoint(ssd300_eval, args.checkpoint)

    evaluator = AsyncEvaluator(num_threads=1)

    coco_eval(args,
              ssd300_eval,
              val_loader,
              cocoGt,
              encoder,
              inv_map,
              0, # epoch
              0, # iter_num
              evaluator=evaluator)


if __name__ == "__main__":
    args = parse_args()
    validate_arguments(args)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)

    run_eval(args)



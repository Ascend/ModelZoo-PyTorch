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

import torch
import os

# DALI import
from .dali_iterator import COCOPipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from box_coder import dboxes300_coco

anchors_ltrb_list = dboxes300_coco()("ltrb").numpy().flatten().tolist()

def prebuild_dali_pipeline(args):
    train_annotate = os.path.join(args.data, "annotations/bbox_only_instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")
    pipe = COCOPipeline(args.batch_size * args.input_batch_multiplier,
                        args.local_rank, train_coco_root,
                        args.meta_files_path, train_annotate, args.N_gpu,
                        anchors_ltrb_list,
                        num_threads=args.num_workers,
                        output_fp16=args.use_fp16, output_nhwc=args.nhwc,
                        pad_output=args.pad_input, seed=args.local_seed - 2**31,
                        use_nvjpeg=args.use_nvjpeg,
                        dali_cache=args.dali_cache,
                        dali_async=(not args.dali_sync))
    pipe.build()
    return pipe

def build_dali_pipeline(args, training=True, pipe=None):
    # pipe is prebuilt without touching the data
    train_loader = DALIGenericIterator(pipelines=[pipe],
                                       output_map= ['image', 'bbox', 'label'],
                                       size=pipe.epoch_size()['train_reader'] // args.N_gpu,
                                       auto_reset=True)
    return train_loader, pipe.epoch_size()['train_reader']

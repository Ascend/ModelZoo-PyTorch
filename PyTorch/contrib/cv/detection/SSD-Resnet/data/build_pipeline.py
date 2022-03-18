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

import torch

from .native_pipeline import build_native_pipeline
#from .dali_pipeline import prebuild_dali_pipeline, build_dali_pipeline####################
from .input_iterators import ConvertDaliInputIterator, RateMatcher, FakeInputIterator
from torch.utils.data import DataLoader
from mlperf_logger import log_event
from mlperf_logging.mllog import constants

"""
Build a train pipe for training (without touching the data)

returns train_pipe
"""
def prebuild_pipeline(args):

    return None

"""
Build a data pipeline for either training or eval

Training : returns loader, epoch_size
Eval : returns loader, inv_class_map, cocoGt
"""
def build_pipeline(args, training=True, pipe=None):
    # Handle training / testing differently due to different
    # outputs. But still want to do this to abstract out the
    # use of EncodingInputIterator and RateMatcher
    if training:
        builder_fn = build_native_pipeline

        train_loader, epoch_size = builder_fn(args, training=True, pipe=pipe)
        log_event(key=constants.TRAIN_SAMPLES, value=epoch_size)

        if args.fake_input:
            train_loader = FakeInputIterator(train_loader, epoch_size, args.N_gpu)

        if args.input_batch_multiplier > 1:
            train_loader = RateMatcher(input_it=train_loader, output_size=args.batch_size)

        return train_loader, epoch_size
    else:
        return build_native_pipeline(args, training=False)

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import ctypes
import logging

import numpy as np
from itertools import accumulate

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import time

# Defines the pipeline for a single GPU for _training_
class COCOPipeline(Pipeline):
    def __init__(self, batch_size, device_id, file_root, meta_files_path, annotations_file, num_gpus,
                 anchors_ltrb_list,
                 output_fp16=False, output_nhwc=False, pad_output=False, num_threads=1,
                 seed=15, dali_cache=-1, dali_async=True,
                 use_nvjpeg=False):
        super(COCOPipeline, self).__init__(batch_size=batch_size, device_id=device_id,
                                           num_threads=num_threads, seed = seed,
                                           exec_pipelined=dali_async,
                                           exec_async=dali_async)

        self.use_nvjpeg = use_nvjpeg
        try:
            shard_id = torch.distributed.get_rank()
        # Note: <= 19.05 was a RuntimeError, 19.06 is now throwing AssertionError
        except (RuntimeError, AssertionError):
            shard_id = 0

        if meta_files_path == None:
            self.c_input = ops.COCOReader(
                file_root = file_root,
                annotations_file = annotations_file,
                shard_id = shard_id,
                num_shards = num_gpus,
                ratio=True,
                ltrb=True,
		#bbox_layout="xyXY",
                skip_empty = True,
                random_shuffle=(dali_cache>0),
                stick_to_shard=(dali_cache>0),
                lazy_init=True,
                shuffle_after_epoch=(dali_cache<=0))
        else:
            self.c_input = ops.COCOReader(
                file_root = file_root,
                meta_files_path = meta_files_path,
                shard_id = shard_id,
                num_shards = num_gpus,
                random_shuffle=(dali_cache>0),
                stick_to_shard=(dali_cache>0),
                lazy_init=True,
                shuffle_after_epoch=(dali_cache<=0))

        self.c_crop = ops.RandomBBoxCrop(device="cpu",
                                       aspect_ratio=[0.5, 2.0],
                                       thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                       scaling=[0.3, 1.0],
                                       ltrb=True,
				       #bbox_layout="xyXY",	
                                       allow_no_crop=True,
                                       num_attempts=1)
        decoder_device = 'mixed' if use_nvjpeg else 'cpu'
        # fused decode and slice.  This is "region-of-interest" (roi) decoding
        self.m_decode = ops.ImageDecoderSlice(device=decoder_device, output_type = types.RGB)
        self.g_slice = None

        # special case for using dali decode caching: the caching decoder can't
        # be fused with slicing (because we need to slice the decoded image
        # differently every epoch), so we need to unfuse decode and slice:
        if dali_cache > 0 and use_nvjpeg:
            self.m_decode = ops.ImageDecoder(device='mixed', output_type = types.RGB,
                                           cache_size=dali_cache*1024, cache_type="threshold",
                                           cache_threshold=10000)
            self.g_slice = ops.Slice(device="gpu")

        # Augumentation techniques (in addition to random crop)
        self.g_twist = ops.ColorTwist(device="gpu")

        self.g_resize = ops.Resize(
            device = "gpu",
            resize_x = 300,
            resize_y = 300,
            min_filter = types.DALIInterpType.INTERP_TRIANGULAR)

        output_dtype = types.FLOAT16 if output_fp16 else types.FLOAT
        output_layout = types.NHWC if output_nhwc else types.NCHW

        mean_val = list(np.array([0.485, 0.456, 0.406]) * 255.)
        std_val = list(np.array([0.229, 0.224, 0.225]) * 255.)
        self.g_normalize = ops.CropMirrorNormalize(device="gpu", crop=(300, 300),
                                                 mean=mean_val,
                                                 std=std_val,
                                                 output_dtype=output_dtype,
                                                 output_layout=output_layout,
                                                 pad_output=pad_output)

        # Random variables
        self.c_rng1 = ops.Uniform(range=[0.5, 1.5])
        self.c_rng2 = ops.Uniform(range=[0.875, 1.125])
        self.c_rng3 = ops.Uniform(range=[-0.5, 0.5])

        flip_probability = 0.5
        self.c_flip_coin = ops.CoinFlip(probability=flip_probability) # coin_rnd

        self.c_bbflip = ops.BbFlip(device="cpu", ltrb=True)

        self.g_box_encoder = ops.BoxEncoder(
            device="gpu",
            criteria=0.5,
            anchors=anchors_ltrb_list,
            offset=True,
            stds=[0.1, 0.1, 0.2, 0.2],
            scale=300)

        self.g_cast = ops.Cast(device="gpu", dtype=types.FLOAT)

    def define_graph(self):
        c_saturation = self.c_rng1()
        c_contrast = self.c_rng1()
        c_brightness = self.c_rng2()
        c_hue = self.c_rng3()
        c_flip = self.c_flip_coin()

        c_inputs, c_bboxes, c_labels = self.c_input(name='train_reader')

        c_crop_begin, c_crop_size, c_bboxes, c_labels = self.c_crop(c_bboxes, c_labels)

        c_bboxes = self.c_bbflip(c_bboxes, horizontal=c_flip)

        if self.g_slice is None:
            g_images = self.m_decode(c_inputs, c_crop_begin, c_crop_size).gpu()
        else:
            g_images = self.m_decode(c_inputs).gpu()
            g_images = self.g_slice(g_images, c_crop_begin, c_crop_size)

        g_images = self.g_resize(g_images)
        g_images = self.g_twist(g_images, saturation=c_saturation, contrast=c_contrast, brightness=c_brightness, hue=c_hue)
        g_images = self.g_normalize(g_images, mirror=c_flip)

        g_labels = c_labels.gpu()
        g_bboxes = c_bboxes.gpu()
        g_bboxes, g_labels = self.g_box_encoder(g_bboxes, g_labels)
        g_labels = self.g_cast(g_labels)

        # bboxes and images and labels on GPU
        return (g_images, g_bboxes, g_labels)

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}

def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to pyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr

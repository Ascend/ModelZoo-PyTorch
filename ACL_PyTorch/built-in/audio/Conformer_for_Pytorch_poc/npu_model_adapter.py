# Copyright 2023 Huawei Technologies Co., Ltd
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
import functools
import numpy as np
from espnet_onnx.utils.function import (
    make_pad_mask,
    mask_fill
)
from ais_bench.infer.interface import InferSession
import time


######################################
# Inference interface for Ascned NPU #
######################################
class AscendInferSession:
    def __init__(self, device_id, config, mode='dymshape', model_path=None, fp16=False):
        self.mode = mode
        if mode not in ["static", "dymdims", "dymshape"]:
            raise NotImplementedError(f"Only support: static/dymshape/dymdims, but now: {self.mode}")
        self.config = config
        if model_path is None:
            model_path = config.model_path
        self.model = InferSession(device_id, model_path)
        self.fp16 = fp16
        self.time = 0

    def run_sequence(self, datas):
        if self.fp16:
            for idx, d in enumerate(datas):
                if d.dtype == np.float32:
                    datas[idx] = d.astype("float16")
        if self.mode == "static":
            st = time.time()
            result =  self.model.infer(datas)
            et = time.time()
        else:
            st = time.time()
            result = self.model.infer(datas,
                                    mode=self.mode,
                                    custom_sizes=self.config.output_size)
            et = time.time()
        self.time += et - st
        return result

    def run(self, out_names, input_dict):
        datas = list(input_dict.values())
        return self.run_sequence(datas)

    def get_inputs(self):
        return self.model.get_inputs()

    def get_outputs(self):
        return self.model.get_outputs()


####################################
# Added function for npu inference #
####################################
def _update_paras(self, remined_idxes):
    self.batch = len(remined_idxes)
    self.end_frames = np.array([self.ori_end_frames[idx] for idx in remined_idxes])
    self.x = self.ori_x[:, :, remined_idxes, :]
    self.idx_b = self.idx_b[:self.batch]
    self.idx_bo = self.idx_bo[:self.batch]


def ctcprefixscorer_npu_init(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        providers = args[3]
        if providers[0] == 'NPUExecutionProvider':
            self = args[0]
            ctc = args[1]
            eos = args[2]
            self.ctc = AscendInferSession(kwargs['device_id'], ctc, fp16=kwargs['fp16'])
            self.eos = eos
            self.impl = None
        else:
            func(*args, **kwargs)
    return wrapper


def ctcprefixscoreth_npu_init(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'multi_batch' in kwargs:
            self = args[0]
            del kwargs['multi_batch']
            func(*args, **kwargs)
            self.ori_end_frames = self.end_frames.copy()
            self.ori_x = self.x.copy()
        else:
            func(*args, **kwargs)
    return wrapper


def ctcprefixscoreth_npu_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'remined_list' in kwargs:
            self = args[0]
            remined_list = kwargs['remined_list']
            if remined_list is not None and len(remined_list) != self.batch:
                self.batch = len(remined_list)
                self.end_frames = np.array([self.ori_end_frames[idx] for idx in remined_list])
                self.x = self.ori_x[:, :, remined_list, :]
                self.idx_b = self.idx_b[:self.batch]
                self.idx_bo = self.idx_bo[:self.batch]
            del kwargs['remined_list']
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


#################################
# Decorator for NPU model class #
#################################
def encoder_npu_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.providers[0] == 'NPUExecutionProvider':
            datas = args[1]
            # multi batch mode
            feats, feat_length, mask, pos_mask, conv_mask, encoder_out_lens = datas
            outputs = self.encoder.run_sequence([feats, mask, pos_mask, conv_mask])
            encoder_out = outputs[0]

            if self.config.enc_type == 'RNNEncoder':
                encoder_out = mask_fill(encoder_out, make_pad_mask(
                    feat_length, encoder_out, 1), 0.0)
            encoder_out = self.mask_output(encoder_out, encoder_out_lens)
            return encoder_out, encoder_out_lens
        else:
            return func(*args, **kwargs)
    return wrapper

def build_encoder_npu_model(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'use_npu' in kwargs:
            self = args[0]
            self.config = args[1]
            self.providers = args[2]
            use_quantized = args[3]
            self.rank_mode = kwargs['rank_mode']
            self.disable_preprocess = kwargs['disable_preprocess']
            if self.rank_mode:
                self.encoder = AscendInferSession(kwargs['device_id'], self.config, fp16=kwargs['fp16'], mode='dymdims')
            else:
                self.encoder = AscendInferSession(kwargs['device_id'], self.config, fp16=kwargs['fp16'])
        else:
            func(*args, **kwargs)
    return wrapper


def build_decoder_npu_model(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'use_npu' in kwargs:
            self = args[0]
            self.config = args[1]
            self.decoder = AscendInferSession(kwargs['device_id'], self.config, fp16=kwargs['fp16'])
            self.n_layers = self.config.n_layers
            self.odim = self.config.odim
            self.in_caches = [d.name for d in self.decoder.get_inputs()
                              if 'cache' in d.name]
            self.out_caches = [d.name for d in self.decoder.get_outputs()
                               if 'cache' in d.name]
        else:
            func(*args, **kwargs)
    return wrapper


def build_joint_network_npu_model(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'use_npu' in kwargs:
            self = args[0]
            self.config = args[1]
            self.joint_session = AscendInferSession(kwargs['device_id'], self.config, fp16=kwargs['fp16'])
        else:
            func(*args, **kwargs)
    return wrapper


def build_transformer_lm_npu_model(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'use_npu' in kwargs:
            self = args[0]
            self.config = args[1]
            self.lm_session = AscendInferSession(kwargs['device_id'], self.config, fp16=kwargs['fp16'])
            self.enc_output_names = ['y'] \
                + [d.name for d in self.lm_session.get_outputs() if 'cache' in d.name]
            self.enc_in_cache_names = [
                d.name for d in self.lm_session.get_inputs() if 'cache' in d.name]

            self.nlayers = self.config.nlayers
            self.odim = self.config.odim
        else:
            func(*args, **kwargs)
    return wrapper

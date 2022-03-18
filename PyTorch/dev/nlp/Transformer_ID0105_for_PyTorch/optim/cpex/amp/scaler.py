# Copyright (c) 2020, Huawei Technologies.
# Copyright (c) 2019, NVIDIA CORPORATION.
# All rights reserved.
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

import torch
import torch.distributed as dist
from apex.multi_tensor_apply import multi_tensor_applier
from ._amp_state import _amp_state, master_params, maybe_print
from itertools import product
import importlib

def scale_check_overflow_python(model_grad, master_grad, scale, check_overflow=False):
    # Exception handling for 18.04 compatibility
    if check_overflow:
        overflow_flag = LossScaler.get_npu_overflow_flag()
        if overflow_flag:
            return True

    if master_grad is not model_grad: # copy_ probably internally short-circuits this
        master_grad.copy_(model_grad)
    if scale != 1.0:
        master_grad.mul_(scale)
    return False

def axpby_check_overflow_python(model_grad, stashed_grad, master_grad, a, b, use_npu_fused_optimizer, 
                                check_overflow=False):
    # Exception handling for 18.04 compatibility
    if check_overflow:
        overflow_flag = LossScaler.get_npu_overflow_flag()
        if overflow_flag:
            return True

    # if master_grad is not model_grad: # copy_ probably internally short-circuits this
    #     master_grad.copy_(model_grad)
    assert stashed_grad.dtype == master_grad.dtype
    converted_model_grad = model_grad.data.to(master_grad.dtype)
    if use_npu_fused_optimizer:
        master_grad.data[:] = a*converted_model_grad.data + b*stashed_grad.data
    else:
        master_grad.data = a*converted_model_grad.data + b*stashed_grad.data
    return False

class LossScaler(object):
    warned_no_fused_kernel = False
    warned_unscaling_non_fp32_grad = False
    has_fused_kernel = False
    npu_float_status = None

    def __init__(self,
                 loss_scale,
                 init_scale=2.**16,
                 scale_factor=2.,
                 scale_window=2000,
                 min_loss_scale=None,
                 max_loss_scale=2.**24):
        if loss_scale == "dynamic":
            self.dynamic = True
            self._loss_scale = min(max_loss_scale, init_scale)
            try:
                LossScaler.npu_float_status = importlib.import_module("npu_float_status")
            except Exception as e:
                maybe_print(
                    '\nImport module "npu_float_status" failed, '
                    'please install apex with --global-option="--npu_float_status" and then try again!\n')
                raise(e)
        else:
            self.dynamic = False
            self._loss_scale = loss_scale
            LossScaler.npu_float_status = None
        self._max_loss_scale = max_loss_scale
        self._min_loss_scale = min_loss_scale
        self._scale_seq_len = scale_window
        self._unskipped = 0
        self._has_overflow = False
        self._overflow_buf = torch.Tensor([0.]).to('npu')
        self._dist_initialized = False

        if dist.is_initialized():
            self._dist_initialized = True

        if multi_tensor_applier.available:
            import amp_C
            LossScaler.has_fused_kernel = multi_tensor_applier.available
            LossScaler.multi_tensor_scale_cuda = amp_C.multi_tensor_scale
            LossScaler.multi_tensor_axpby_cuda = amp_C.multi_tensor_axpby
        else:
            LossScaler.has_fused_kernel = False
            LossScaler.warned_no_fused_kernel = True

    @staticmethod
    def get_npu_overflow_flag():
        if LossScaler.npu_float_status is not None:
            return LossScaler.npu_float_status.RunGetFloatStatusOp()
        else:
            return False
    
    @staticmethod
    def clear_npu_overflow_flag():
        if LossScaler.npu_float_status is not None:
            LossScaler.npu_float_status.RunClearFloatStatusOp()

    def loss_scale(self):
        return self._loss_scale

    def unscale_python(self, model_grads, master_grads, scale):
        for model, master in zip(model_grads, master_grads):
            if model is not None:
                if not LossScaler.warned_unscaling_non_fp32_grad:
                    if master.dtype != torch.float32:
                        maybe_print(
                            "Attempting to unscale a grad with type {} ".format(master.type()) +
                            "Unscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_unscaling_non_fp32_grad = True
                self._has_overflow = scale_check_overflow_python(model,
                                                                 master,
                                                                 1./scale,
                                                                 self.dynamic)
                if self._has_overflow and self.dynamic:
                    break

        if self._has_overflow:
            if self.dynamic and self._dist_initialized:
                self._overflow_buf.add_(1)
                dist.all_reduce(self._overflow_buf)
                self._overflow_buf.zero_()
        else:
            if self.dynamic and self._dist_initialized:
                dist.all_reduce(self._overflow_buf)
                if self._overflow_buf.item() != 0:
                    self._has_overflow = True
                self._overflow_buf.zero_()

    # unused_scale keeps some of the old API alive for hopefully a short time.
    def unscale(self, model_grads, master_grads, unused_scale, models_are_masters=False, scale_override=None):
        if self._has_overflow:
            return

        scale = self._loss_scale
        if scale_override is not None:
            scale = scale_override

        if scale == 1.0 and models_are_masters and not self.dynamic:
            return

        if LossScaler.has_fused_kernel:
            # if (not LossScaler.warned_unscaling_non_fp32_grad
            #     and master_grads[0].dtype == torch.float16):
            #     print("Warning:  unscaling grads that are not FP32. "
            #           "Unscaling non-fp32 grads may indicate an error. "
            #           "When using Amp, you don't need to call .half() on your model.")
            #     # Setting this to True unconditionally allows the possibility of an escape
            #     # if never-before-seen non-fp32 grads are created in some later iteration.
            #     LossScaler.warned_unscaling_non_fp32_grad = True
            multi_tensor_applier(LossScaler.multi_tensor_scale_cuda,
                                 self._overflow_buf,
                                 [model_grads, master_grads],
                                 1./scale)
        else:
            self.unscale_python(model_grads, master_grads, scale)

    def unscale_with_stashed_python(self,
                                    model_grads,
                                    stashed_master_grads,
                                    master_grads,
                                    a,
                                    b,
                                    use_npu_fused_optimizer):
        for model, stashed, master in zip(model_grads, stashed_master_grads, master_grads):
            if model is None and stashed is None:
                continue
            else:
                if not LossScaler.warned_unscaling_non_fp32_grad:
                    if master.dtype != torch.float32:
                        maybe_print(
                            "Attempting to unscale a grad with type {} ".format(master.type()) +
                            "Unscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_unscaling_non_fp32_grad = True
                self._has_overflow = axpby_check_overflow_python(model,
                                                                 stashed,
                                                                 master,
                                                                 a,
                                                                 b,
                                                                 use_npu_fused_optimizer,
                                                                 self.dynamic)
                if self._has_overflow and self.dynamic:
                    break

        if self._has_overflow:
            if self.dynamic and self._dist_initialized:
                self._overflow_buf.add_(1)
                dist.all_reduce(self._overflow_buf)
                self._overflow_buf.zero_()
        else:
            if self.dynamic and self._dist_initialized:
                dist.all_reduce(self._overflow_buf)
                if self._overflow_buf.item() != 0:
                    self._has_overflow = True
                self._overflow_buf.zero_()

    def unscale_with_stashed(self,
                             model_grads,
                             stashed_master_grads,
                             master_grads,
                             scale_override=None,
                             use_npu_fused_optimizer=False):
        if self._has_overflow:
            return

        grads_have_scale, stashed_have_scale, out_scale = self._loss_scale, 1.0, 1.0
        if scale_override is not None:
            grads_have_scale, stashed_have_scale, out_scale = scale_override

        if LossScaler.has_fused_kernel:
            if (not LossScaler.warned_unscaling_non_fp32_grad
                and master_grads[0].dtype == torch.float16):
                print("Warning:  unscaling grads that are not FP32. "
                      "Unscaling non-fp32 grads may indicate an error. "
                      "When using Amp, you don't need to call .half() on your model.")
                # Setting this to True unconditionally allows the possibility of an escape
                # if never-before-seen non-fp32 grads are created in some later iteration.
                LossScaler.warned_unscaling_non_fp32_grad = True
            multi_tensor_applier(LossScaler.multi_tensor_axpby_cuda,
                                 self._overflow_buf,
                                 [model_grads, stashed_master_grads, master_grads],
                                 out_scale/grads_have_scale,   # 1./scale,
                                 out_scale/stashed_have_scale, # 1.0,
                                 0) # check only arg 0, aka the incoming model grads, for infs
        else:
            self.unscale_with_stashed_python(model_grads,
                                             stashed_master_grads,
                                             master_grads,
                                             out_scale/grads_have_scale,
                                             out_scale/stashed_have_scale,
                                             use_npu_fused_optimizer)
    def unscale_with_stashed_combined(self,
                                      grads_combined,
                                      stashed_grads_combined,
                                      scale_override=None):

        if self.dynamic:
            self._has_overflow = LossScaler.get_npu_overflow_flag()
            if self._dist_initialized:
                if self._has_overflow:
                    self._overflow_buf.add_(1)
                    dist.all_reduce(self._overflow_buf)
                    self._overflow_buf.zero_()
                else:
                    dist.all_reduce(self._overflow_buf)
                    if self._overflow_buf.item() != 0:
                        self._has_overflow = True
                    self._overflow_buf.zero_()

        if self._has_overflow:
            return
        
        grads_have_scale, stashed_have_scale, out_scale = self._loss_scale, 1.0, 1.0
        if scale_override is not None:
            grads_have_scale, stashed_have_scale, out_scale = scale_override

        # TODO convert dtype in O2
        grads_combined.data[:] = grads_combined.mul_(out_scale/grads_have_scale) + stashed_grads_combined

        # Defer to update_scale
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        # if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        #     self._has_overflow = self._overflow_buf.item()
    def unscale_grad_O2(self,
                        model_grads_combined=None,
                        stashed_master_grads_combined=None,
                        master_grads_combined=None,
                        scale_override=None,
                        master_grads=None,
                        model_grads=None):

        if self.dynamic:
            self._has_overflow = LossScaler.get_npu_overflow_flag()
            if self._dist_initialized:
                if self._has_overflow:
                    self._overflow_buf.add_(1)
                    dist.all_reduce(self._overflow_buf)
                    self._overflow_buf.zero_()
                else:
                    dist.all_reduce(self._overflow_buf)
                    if self._overflow_buf.item() != 0:
                        self._has_overflow = True
                    self._overflow_buf.zero_()

        if self._has_overflow:
            return
        
        grads_have_scale, stashed_have_scale, out_scale = self._loss_scale, 1.0, 1.0
        if scale_override is not None:
            grads_have_scale, stashed_have_scale, out_scale = scale_override


        if stashed_master_grads_combined != None and master_grads_combined.data_ptr() == stashed_master_grads_combined.data_ptr() and master_grads_combined.numel() == stashed_master_grads_combined.numel():
            stashed_master_grads_combined = master_grads_combined.clone()

        if master_grads_combined is not model_grads_combined:
            if master_grads_combined.numel() == model_grads_combined.numel():
                master_grads_combined.copy_(model_grads_combined)
            else:
                for master, model in zip(master_grads, model_grads):
                    master.copy_(model)
        master_grads_combined.mul_(out_scale/grads_have_scale)

        if stashed_master_grads_combined is not None:
            assert stashed_master_grads_combined.dtype == master_grads_combined.dtype
            master_grads_combined.add_(stashed_master_grads_combined)

    def clear_overflow_state(self):
        self._has_overflow = False
        if self.has_fused_kernel:
            self._overflow_buf.zero_()

    # Separate so unscale() can be called more that once before updating.
    def update_scale(self):
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

        if self._has_overflow and self.dynamic:
            should_skip = True
            if(self._min_loss_scale):
                self._loss_scale = max(self._min_loss_scale, self._loss_scale/2.)
            else:
                self._loss_scale = self._loss_scale/2.
            self._unskipped = 0
        else:
            should_skip = False
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len and self.dynamic:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale*2.)
            self._unskipped = 0

        return should_skip

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

import types
from apex.fp16_utils import master_params_to_model_params
from apex.multi_tensor_apply import multi_tensor_applier
from ._amp_state import maybe_print
import torch
from apex.optimizers import FusedSGD, NpuFusedAdam, NpuFusedSGD, NpuFusedAdadelta
from change_data_ptr import change_data_ptr
from apex.contrib.combine_tensors import combine_npu


def get_grad_combined_tensor_from_param(list_of_params):
    if len(list_of_params) > 0 and list_of_params[0].grad is not None:
        list_of_grad = []
        for param in list_of_params:
            if param.requires_grad:
                list_of_grad.append(param.grad)
        original_combined_tensor = combine_npu(list_of_grad)
        return original_combined_tensor, list_of_grad
    else:
        return None, []


class AmpOptimizerState(object):
    def __init__(self):
        pass


def _master_params_to_model_params(self):
    stash = self._amp_stash
    if multi_tensor_applier.available:
        if len(stash.all_fp16_params) > 0:
            multi_tensor_applier(
                stash.multi_tensor_scale,
                stash.dummy_overflow_buf,
                [stash.all_fp32_from_fp16_params, stash.all_fp16_params],
                1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(stash.fp16_groups, stash.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)


def lazy_init_with_master_weights(self):
    stash = self._amp_stash
    stash.fp16_groups = []
    stash.fp32_from_fp16_groups = []
    stash.fp32_from_fp32_groups = []
    for i, param_group in enumerate(self.param_groups):
        # maybe_print("FP16_Optimizer processing param group {}:".format(i))
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.npu.HalfTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                    #             .format(param.size()))
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    param_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                    # Reset existing state dict key to the new master param.
                    # We still need to recast per-param state tensors, if any, to FP32.
                    if param in self.state:
                        self.state[master_param] = self.state.pop(param)
                elif param.type() == 'torch.npu.FloatTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                    #             .format(param.size()))
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError("Optimizer's parameters must be either "
                                    "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                    "Received {}".format(param.type()))

        stash.fp16_groups.append(fp16_params_this_group)
        stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        stash.fp32_from_fp32_groups.append(fp32_params_this_group)

    stash.all_fp16_params = []
    for group in stash.fp16_groups:
        stash.all_fp16_params += group

    stash.all_fp32_from_fp16_params = []
    for group in stash.fp32_from_fp16_groups:
        stash.all_fp32_from_fp16_params += group

    stash.all_fp32_from_fp32_params = []
    for group in stash.fp32_from_fp32_groups:
        stash.all_fp32_from_fp32_params += group

    # all_fp16_grad_stash is only needed for fused optimizers.
    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    # stash.all_fp32_from_fp16_grad_stash = [None for _ in stash.all_fp32_from_fp16_params]
    stash.all_fp32_from_fp32_grad_stash = [None for _ in stash.all_fp32_from_fp32_params]
    if self.accelerate:
        for param in stash.all_fp32_from_fp16_params:
            param.grad = None
    else:
        for param in stash.all_fp32_from_fp16_params:
            param.grad = None

    for param in stash.all_fp32_from_fp32_params:
        param.grad = None
    
    if self.accelerate:
        stash.combined_tensor_fp16 = None
        stash.combined_tensor_fp32_from_fp16 = None
        stash.combined_tensor_fp32 = None

        stash.combined_tensor_fp32_grad_stash = None
    # Leverage state_dict() and load_state_dict() to recast preexisting per-param state tensors
    self.load_state_dict(self.state_dict())


def post_backward_models_are_masters(scaler, params, stashed_grads, scale_override=None, 
                                     main_grads_combined=None, stashed_grads_combined=None, 
                                     use_npu_fused_optimizer=False):
    grads_have_scale, stashed_have_scale, out_scale = scaler.loss_scale(), 1.0, 1.0

    # not much to do if scale == 1.0 and static scaling
    if scaler.loss_scale() == 1.0 and not scaler.dynamic:
        # Clear the stash.
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None
        return

    if scale_override is not None:
        grads_have_scale, stashed_have_scale, out_scale = scale_override

    # This is a lot of python overhead...
    if main_grads_combined is not None:
        scaler.unscale_with_stashed_combined(
            main_grads_combined, stashed_grads_combined,
            scale_override=(grads_have_scale, stashed_have_scale, out_scale))
    else:
        grads_needing_unscale = []
        grads_needing_unscale_with_stash = []
        stashed = []
        for param, stashed_grad in zip(params, stashed_grads):
            if param.grad is None and stashed_grad is not None:
                param.grad = stashed_grad
            elif param.grad is not None and stashed_grad is None:
                grads_needing_unscale.append(param.grad)
            elif param.grad is not None and stashed_grad is not None:
                grads_needing_unscale_with_stash.append(param.grad)
                stashed.append(stashed_grad)
            else:  # param.grad is None and stashed_grad is None
                continue

        # unscale() implements grads*(1/scale), so "scale" should be grads_have_scale/out_scale.
        if len(grads_needing_unscale) > 0:
            scaler.unscale(
                grads_needing_unscale,
                grads_needing_unscale,
                None,  # unused_scale, currently present to avoid API breakage elsewhere
                models_are_masters=True,
                scale_override=grads_have_scale / out_scale)

        if len(grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(
                grads_needing_unscale_with_stash,
                stashed,
                grads_needing_unscale_with_stash,
                scale_override=(grads_have_scale, stashed_have_scale, out_scale),
                use_npu_fused_optimizer=use_npu_fused_optimizer)

        # Clear the stash.
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None


def prepare_backward_with_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()

    if self.accelerate:
        if stash.process_zero_grad:
            return
        if stash.combined_tensor_fp32 is not None:
            stash.combined_tensor_fp32_grad_stash.copy_(stash.combined_tensor_fp32)
            stash.combined_tensor_fp32.zero_()
    else:
        for i, param in enumerate(stash.all_fp16_params):
            param.grad = None

        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            if self.is_npu_fused_optimizer:
                if param.grad is not None:
                    stash.all_fp32_from_fp32_grad_stash[i] = param.grad.clone()
                    param.grad.zero_()
                else:
                    stash.all_fp32_from_fp32_grad_stash[i] = None
            else:
                stash.all_fp32_from_fp32_grad_stash[i] = param.grad
                # Set up to leverage grad copy elision:
                param.grad = None

def combined_init_with_master_weights(stash):
    if not stash.already_combined:
        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            if param.grad is not None:
                stash.all_fp32_from_fp32_grad_stash[i] = param.grad.clone()

        if len(stash.all_fp32_from_fp32_grad_stash) > 0:
            stash.combined_tensor_fp32_grad_stash = combine_npu(stash.all_fp32_from_fp32_grad_stash)

        stash.combined_tensor_fp16, stash.fp16_param_grad_list = get_grad_combined_tensor_from_param(stash.all_fp16_params)
        for model_grad, master in zip(stash.fp16_param_grad_list, stash.all_fp32_from_fp16_params):
            master.grad = torch.empty_like(model_grad.to(torch.float))
            master.data = master.data.npu_format_cast(model_grad.storage().npu_format())

        stash.combined_tensor_fp32_from_fp16, stash.fp32_from_fp16_param_grad_list = get_grad_combined_tensor_from_param(stash.all_fp32_from_fp16_params)
        stash.combined_tensor_fp32, stash.fp32_param_grad_list = get_grad_combined_tensor_from_param(stash.all_fp32_from_fp32_params)
        # please do not change the order of tensor in this list.
        stash.grads_list = [stash.combined_tensor_fp16, stash.combined_tensor_fp32_from_fp16, stash.combined_tensor_fp32]
        stash.already_combined = True


def post_backward_with_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    if self.accelerate:
        combined_init_with_master_weights(stash)
        scaler.unscale_grad_O2(
            model_grads_combined=stash.combined_tensor_fp16,
            stashed_master_grads_combined=stash.combined_tensor_fp32_from_fp16 if not stash.process_zero_grad else None,
            master_grads_combined=stash.combined_tensor_fp32_from_fp16,
            master_grads=stash.fp32_from_fp16_param_grad_list,
            model_grads=stash.fp16_param_grad_list)
        if stash.combined_tensor_fp32 is not None:
            scaler.unscale_grad_O2(
                model_grads_combined=stash.combined_tensor_fp32,
                stashed_master_grads_combined=stash.combined_tensor_fp32_grad_stash if not stash.process_zero_grad else None,
                master_grads_combined=stash.combined_tensor_fp32)
        stash.process_zero_grad = False
    else:
        # This is a lot of python overhead...
        fp16_grads_needing_unscale = []
        new_fp32_grads = []
        fp16_grads_needing_unscale_with_stash = []
        preexisting_fp32_grads = []
        for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                          stash.all_fp32_from_fp16_params):
            if fp16_param.grad is None and fp32_param.grad is not None:
                continue
            elif fp16_param.grad is not None and fp32_param.grad is None:
                fp32_param.grad = torch.empty_like(fp32_param)
                fp16_grads_needing_unscale.append(fp16_param.grad)
                new_fp32_grads.append(fp32_param.grad)
            elif fp16_param.grad is not None and fp32_param.grad is not None:
                if stash.all_fp32_from_fp16_params_grad_is_zero:
                    fp16_grads_needing_unscale.append(fp16_param.grad)
                    new_fp32_grads.append(fp32_param.grad)
                else:
                    fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
                    preexisting_fp32_grads.append(fp32_param.grad)
            else: # fp16_param.grad is None and fp32_param.grad is None:
                continue

        if len(fp16_grads_needing_unscale) > 0:
            scaler.unscale(
                fp16_grads_needing_unscale,
                new_fp32_grads,
                scaler.loss_scale(),
                models_are_masters=False)

        if len(fp16_grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(
                fp16_grads_needing_unscale_with_stash,
                preexisting_fp32_grads,
                preexisting_fp32_grads,
                use_npu_fused_optimizer=self.is_npu_fused_optimizer)

        stash.all_fp32_from_fp16_params_grad_is_zero = False

        # fp32 params can be treated as they would be in the "no_master_weights" case.
        post_backward_models_are_masters(
            scaler,
            stash.all_fp32_from_fp32_params,
            stash.all_fp32_from_fp32_grad_stash,
            use_npu_fused_optimizer=self.is_npu_fused_optimizer)


def lazy_init_no_master_weights(self):
    stash = self._amp_stash
    stash.all_fp16_params = []
    stash.all_fp32_params = []

    for i, param_group in enumerate(self.param_groups):
        for i, param in enumerate(param_group['params']):
            if param.type() == 'torch.npu.HalfTensor':
                stash.all_fp16_params.append(param)
            elif param.type() == 'torch.npu.FloatTensor':
                stash.all_fp32_params.append(param)
            else:
                raise TypeError("Optimizer's parameters must be either "
                                "torch.npu.FloatTensor or torch.npu.HalfTensor."
                                "Received {}".format(param.type()))

    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    stash.all_fp32_grad_stash = [None for _ in stash.all_fp32_params]

    if self.accelerate:
        stash.all_fp16_grad_stash_combine = None
        stash.all_fp32_grad_stash_combine = None

        stash.fp16_grad_list = []
        stash.main_fp16_grad_combine = None

        stash.fp32_grad_list = []
        stash.main_fp32_grad_combine = None



def prepare_backward_no_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()
    if self.accelerate and stash.already_combined:
        if stash.main_fp16_grad_combine is not None:
            stash.all_fp16_grad_stash_combine.copy_(stash.main_fp16_grad_combine)
            stash.main_fp16_grad_combine.zero_()
        if stash.main_fp32_grad_combine is not None:
            stash.all_fp32_grad_stash_combine.copy_(stash.main_fp32_grad_combine)
            stash.main_fp32_grad_combine.zero_()
    elif self.is_npu_fused_optimizer:
        for i, param in enumerate(stash.all_fp16_params):
            if param.grad is not None:
                stash.all_fp16_grad_stash[i] = param.grad.clone()
                param.grad.zero_()
            else:
                stash.all_fp16_grad_stash[i] = None

        for i, param in enumerate(stash.all_fp32_params):
            if param.grad is not None:
                stash.all_fp32_grad_stash[i] = param.grad.clone()
                param.grad.zero_()
            else:
                stash.all_fp32_grad_stash[i] = None
    else:
        for i, param in enumerate(stash.all_fp16_params):
            stash.all_fp16_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None

        for i, param in enumerate(stash.all_fp32_params):
            stash.all_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def post_backward_no_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()
    if self.accelerate:
        self._amp_combined_init()
        split_types = ((stash.main_fp16_grad_combine, stash.all_fp16_grad_stash_combine),
                (stash.main_fp32_grad_combine, stash.all_fp32_grad_stash_combine))
        for main_grads_combined, stash_grads_combined  in split_types:
            if main_grads_combined is not None:
                post_backward_models_are_masters(scaler, None, None, None, 
                                                 main_grads_combined, stash_grads_combined,
                                                 use_npu_fused_optimizer=self.is_npu_fused_optimizer)
    else:
        split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
                 (stash.all_fp32_params, stash.all_fp32_grad_stash))

        for params, stashed_grads in split_types:
            post_backward_models_are_masters(scaler, params, stashed_grads, 
                                             use_npu_fused_optimizer=self.is_npu_fused_optimizer)


#####################################################################################
# FusedSGD versions
#####################################################################################

# FusedSGD never explicitly materializes the fp32 gradients for "fp32 from fp16" master params
# outside the kernel, so we must accumulate directly into the model grads.
def prepare_backward_with_master_weights_FusedSGD(self):
    if self.materialize_master_grads:
        prepare_backward_with_master_weights(self)
    else:
        stash = self._amp_stash

        self._amp_lazy_init()

        for i, param in enumerate(stash.all_fp16_params):
            stash.all_fp16_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None

        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            stash.all_fp32_from_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def post_backward_with_master_weights_FusedSGD(self, scaler):
    if self.materialize_master_grads:
        post_backward_with_master_weights(self, scaler)
    else:
        stash = self._amp_stash

        self._amp_lazy_init()

        grads_have_scale = scaler.loss_scale()
        stashed_have_scale = self.most_recent_scale
        out_scale = grads_have_scale
        if self.scale_set_by_backward:
            out_scale = min(grads_have_scale, self.most_recent_scale)

        split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
                 (stash.all_fp32_from_fp32_params, stash.all_fp32_from_fp32_grad_stash))


        # unscale_with_stashed() implements grads*1/scale + stashed_grads*1.
        # stashed_grads are scaled by self.most_recent_scale.
        for params, stashed_grads in split_types:
            post_backward_models_are_masters(scaler, params, stashed_grads,
                                             (grads_have_scale, stashed_have_scale, out_scale))

        self.most_recent_scale = out_scale
        self.scale_set_by_backward = True


def prepare_backward_no_master_weights_FusedSGD(self):
    prepare_backward_no_master_weights(self)


def post_backward_no_master_weights_FusedSGD(self, scaler):
    post_backward_no_master_weights(self, scaler)


def _amp_lazy_init(self):
    stash = self._amp_stash

    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True


def combined_init_no_master_weights(self):
    stash = self._amp_stash
    if not stash.already_combined:
        for i, param in enumerate(stash.all_fp16_params):
            if param.grad is not None:
                stash.all_fp16_grad_stash[i] = param.grad.clone()
        for i, param in enumerate(stash.all_fp32_params):
            if param.grad is not None:
                stash.all_fp32_grad_stash[i] = param.grad.clone()

        if len(stash.all_fp16_grad_stash) > 0:
            # if len == 0, avoid to create a useless combined tensor
            stash.all_fp16_grad_stash_combine = combine_npu(stash.all_fp16_grad_stash, require_copy_value=False)
        if len(stash.all_fp32_grad_stash) > 0:
            stash.all_fp32_grad_stash_combine = combine_npu(stash.all_fp32_grad_stash, require_copy_value=False)

        stash.main_fp16_grad_combine, stash.fp16_grad_list = get_grad_combined_tensor_from_param(stash.all_fp16_params)
        stash.main_fp32_grad_combine, stash.fp32_grad_list = get_grad_combined_tensor_from_param(stash.all_fp32_params)
        # please do not change the order of tensor in this list.
        stash.grads_list = [stash.main_fp16_grad_combine, stash.main_fp32_grad_combine]
        stash.already_combined = True

def combine_params_and_grads_by_group(self):
    stash = self._amp_stash
    if stash.params_grads_are_combined:
        return

    for group in self.param_groups:
        params_list = []
        grads_list = []

        for p in group['params']:
            if p.grad is None:
                continue
            params_list.append(p)
            grads_list.append(p.grad)

        params_combined = None
        grads_combined = None

        if len(params_list) > 0:
            params_combined = combine_npu(params_list)
            grads_combined = combine_npu(grads_list)

        stash.params_combined_list.append(params_combined)
        stash.grads_combined_list.append(grads_combined)

    stash.params_grads_are_combined = True

def _process_optimizer(optimizer, properties):
    if hasattr(optimizer, "_amp_stash"):
        raise RuntimeError("A given optimizer should only be passed through amp.initialize once.")
    else:
        optimizer._amp_stash = AmpOptimizerState()

    optimizer._amp_stash.lazy_init_called = False
    optimizer._amp_stash.already_patched = False
    optimizer._amp_stash.params_have_scaled_gradients = False
    optimizer.accelerate = properties.combine_grad
    if optimizer.accelerate:
        optimizer._amp_stash.grads_list = []
    optimizer._amp_stash.already_combined = False

    optimizer._amp_stash.process_zero_grad = True

    optimizer._amp_stash.params_grads_are_combined = False
    optimizer._amp_stash.params_combined_list = []
    optimizer._amp_stash.grads_combined_list = []
    optimizer._amp_stash.all_fp32_from_fp16_params_grad_is_zero = False
    optimizer._amp_stash.param_state_combined_list = []
    optimizer._amp_stash.param_state_combined = False

    for name in ("_lazy_init_maybe_master_weights",
                 "_master_params_to_model_params",
                 "_prepare_amp_backward",
                 "_post_amp_backward",
                 "_amp_lazy_init",
                 "_combine_params_and_grads_by_group"):
        if hasattr(optimizer, name):
            raise RuntimeError("Incoming optimizer already has {} defined.".format(name))

    if isinstance(optimizer, NpuFusedSGD) or isinstance(optimizer, NpuFusedAdam) or \
        isinstance(optimizer, NpuFusedAdadelta):
        optimizer.is_npu_fused_optimizer = True
        optimizer._combine_params_and_grads_by_group = types.MethodType(
            combine_params_and_grads_by_group, optimizer)
    elif hasattr(optimizer, "is_npu_fused_optimizer") and optimizer.is_npu_fused_optimizer:
        optimizer._combine_params_and_grads_by_group = types.MethodType(
            combine_params_and_grads_by_group, optimizer)
    else:
        optimizer.is_npu_fused_optimizer = False

    if optimizer.is_npu_fused_optimizer:
        if properties.opt_level != "O2" or properties.master_weights != True or optimizer.accelerate:
            raise RuntimeError("Currently, npu fused optimizer can only be used when opt_level='O2' "
                               "and master_weights=True and combine_grad=False")

    # TODO:  Centralize exposure and import error checking for the C backend.
    if multi_tensor_applier.available:
        import amp_C
        optimizer._amp_stash.multi_tensor_scale = amp_C.multi_tensor_scale
        optimizer._amp_stash.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        optimizer._amp_stash.dummy_overflow_buf = torch.cuda.IntTensor([0]);

    if properties.master_weights:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_with_master_weights, optimizer)

        optimizer._master_params_to_model_params = types.MethodType(
            _master_params_to_model_params, optimizer)

        old_step = optimizer.step
        def new_step(self, closure=None):
            if closure is not None:
                raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
            retval = old_step()
            if not isinstance(self, FusedSGD):
                self._master_params_to_model_params()
            # Clear the master grads that wouldn't be zeroed by model.zero_grad()
            for param in self._amp_stash.all_fp32_from_fp16_params:
                if (optimizer.is_npu_fused_optimizer or optimizer.accelerate) and param.grad is not None:
                    if optimizer.accelerate:
                        self._amp_stash.combined_tensor_fp32_from_fp16.zero_()
                        break
                    if optimizer.is_npu_fused_optimizer:
                        param.grad.zero_()
                        self._amp_stash.all_fp32_from_fp16_params_grad_is_zero = True
                else:
                    param.grad = None
            return retval
        optimizer.step = types.MethodType(new_step, optimizer)

        old_zero_grad = optimizer.zero_grad
        def new_zero_grad(self):
            stash = self._amp_stash
            self._amp_lazy_init()
            # Zero the model grads.
            stash.process_zero_grad = True
            for param in stash.all_fp16_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            for param in stash.all_fp32_from_fp32_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            # Clear the master grads that are independent of model grads
            for param in self._amp_stash.all_fp32_from_fp16_params:
                if (optimizer.is_npu_fused_optimizer or optimizer.accelerate) and param.grad is not None:
                    param.grad.zero_()
                    if optimizer.is_npu_fused_optimizer:
                        self._amp_stash.all_fp32_from_fp16_params_grad_is_zero = True
                else:
                    param.grad = None
        optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)

        if optimizer.is_npu_fused_optimizer:
            old_load_state_dict = optimizer.load_state_dict
            def new_load_state_dict(self, state_dict):
                old_load_state_dict(state_dict)
                self._amp_stash.param_state_combined = False
            optimizer.load_state_dict = types.MethodType(new_load_state_dict, optimizer)

        if isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights, optimizer)
    else:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_no_master_weights, optimizer)

        if isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights, optimizer)

    optimizer._amp_lazy_init = types.MethodType(_amp_lazy_init, optimizer)
    if optimizer.accelerate:
        optimizer._amp_combined_init = types.MethodType(combined_init_no_master_weights, optimizer)

    old_add_param_group = optimizer.add_param_group

    def new_add_param_group(self, new_group):
        stash = self._amp_stash

        if not stash.lazy_init_called:
            self._lazy_init_maybe_master_weights()
            stash.lazy_init_called = True

        assert isinstance(new_group, dict), "param group must be a dict"

        new_params = new_group['params']
        if isinstance(new_params, torch.Tensor):
            new_group['params'] = [new_params]
        elif isinstance(new_params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            new_group['params'] = list(new_params)

        if properties.master_weights:
            # Mutate new_group in-place to use FP32 master params
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(new_group['params']):
                if param.requires_grad:
                    if param.type() == 'torch.npu.HalfTensor':
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        new_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                    elif param.type() == 'torch.npu.FloatTensor':
                        fp32_params_this_group.append(param)
                        new_group['params'][i] = param
                    else:
                        raise TypeError("Optimizer's parameters must be either "
                                        "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                        "Received {}".format(param.type()))

            stash.fp16_groups.append(fp16_params_this_group)
            stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            stash.fp32_from_fp32_groups.append(fp32_params_this_group)

            stash.all_fp16_params += fp16_params_this_group
            stash.all_fp32_from_fp16_params += fp32_from_fp16_params_this_group
            stash.all_fp32_from_fp32_params += fp32_params_this_group

            # stash.all_fp32_from_fp16_grad_stash = [None for _ in stash.all_fp32_from_fp16_params]
            stash.all_fp32_from_fp32_grad_stash += [None for _ in fp32_params_this_group]

            # It should be ok to let params be added with existing .grad attributes.
            # for param in fp16_params_this_group:
            #     param.grad = None

            # for param in fp32_from_fp16_params_this_group:
            #     param.grad = None

            # for param in stash.fp32_params_this_group:
            #     param.grad = None
        else:
            for param in new_group['params']:
                if param.type() == 'torch.npu.HalfTensor':
                    stash.all_fp16_params.append(param)
                    stash.all_fp16_grad_stash.append(None)
                elif param.type() == 'torch.npu.FloatTensor':
                    stash.all_fp32_params.append(param)
                    stash.all_fp32_grad_stash.append(None)
                else:
                    raise TypeError("Optimizer's parameters must be either "
                                    "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                    "Received {}".format(param.type()))

        old_add_param_group(new_group)

    optimizer.add_param_group = types.MethodType(new_add_param_group, optimizer)

    return optimizer

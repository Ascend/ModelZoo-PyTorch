# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==========================================================================

#!/usr/bin/env python3

"""
Used for EMA tracking a given pytorch module. The user is responsible for calling step()
and setting the appropriate decay
"""

import copy
from dataclasses import dataclass, field
import logging

import torch

from fairseq.dataclass import FairseqDataclass


@dataclass
class EMAModuleConfig(FairseqDataclass):
    ema_decay: float = field(
        default=0.9999, metadata={"help": "decay for exponential moving average model"}
    )
    ema_fp32: bool = field(
        default=False,
        metadata={"help": "If true, store EMA model in fp32 even if model is in fp16"},
    )


class EMAModule:
    """Exponential Moving Average of Fairseq Models"""

    def __init__(self, model, config: EMAModuleConfig, device=None, skip_keys=None):
        """
        @param model model to initialize the EMA with
        @param config EMAConfig object with configuration like
        ema_decay, ema_update_freq, ema_fp32
        @param device If provided, copy EMA to this device (e.g. gpu).
        Otherwise EMA is in the same device as the model.
        """

        self.decay = config.ema_decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.config = config
        self.skip_keys = skip_keys or set()
        self.fp32_params = {}

        if device is not None:
            logging.info(f"Copying EMA model to device {device}")
            self.model = self.model.to(device=device)

        if self.config.ema_fp32:
            self.build_fp32_params()

        self.update_freq_counter = 0

    def build_fp32_params(self, state_dict=None):
        """
        Store a copy of the EMA params in fp32.
        If state dict is passed, the EMA params is copied from
        the provided state dict. Otherwise, it is copied from the
        current EMA model parameters.
        """
        if not self.config.ema_fp32:
            raise RuntimeError(
                "build_fp32_params should not be called if ema_fp32=False. "
                "Use ema_fp32=True if this is really intended."
            )

        if state_dict is None:
            state_dict = self.model.state_dict()

        def _to_float(t):
            return t.float() if torch.is_floating_point(t) else t

        for param_key in state_dict:
            if param_key in self.fp32_params:
                self.fp32_params[param_key].copy_(state_dict[param_key])
            else:
                self.fp32_params[param_key] = _to_float(state_dict[param_key])

    def restore(self, state_dict, build_fp32_params=False):
        """Load data from a model spec into EMA model"""
        self.model.load_state_dict(state_dict, strict=False)
        if build_fp32_params:
            self.build_fp32_params(state_dict)

    def set_decay(self, decay):
        self.decay = decay

    def get_decay(self):
        return self.decay

    def _step_internal(self, new_model):
        """One update of the EMA model based on new model weights"""
        decay = self.decay

        ema_state_dict = {}
        ema_params = (
            self.fp32_params if self.config.ema_fp32 else self.model.state_dict()
        )
        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )
                ema_params[key] = ema_param

            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            if key in self.skip_keys or not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.restore(ema_state_dict, build_fp32_params=False)

    @torch.no_grad()
    def step(self, new_model):
        self._step_internal(new_model)

    def reverse(self, model):
        """
        Load the model parameters from EMA model.
        Useful for inference or fine-tuning from the EMA model.
        """
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]

        model.load_state_dict(d, strict=False)
        return model

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch

from common.text.symbols import get_symbols, get_pad_idx
from fastpitch.model import FastPitch
from fastpitch.model_jit import FastPitchJIT
from waveglow.model import WaveGlow


def parse_model_args(model_name, parser, add_help=False):
    if model_name == 'WaveGlow':
        from waveglow.arg_parser import parse_waveglow_args
        return parse_waveglow_args(parser, add_help)
    elif model_name == 'FastPitch':
        from fastpitch.arg_parser import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, device,
              uniform_initialize_bn_weight=False, forward_is_infer=False,
              jitable=False):

    if model_name == 'WaveGlow':
        model = WaveGlow(**model_config)

    elif model_name == 'FastPitch':
        if jitable:
            model = FastPitchJIT(**model_config)
        else:
            model = FastPitch(**model_config)

    else:
        raise NotImplementedError(model_name)

    if forward_is_infer:
        model.forward = model.infer

    if uniform_initialize_bn_weight:
        init_bn(model)

    return model.to(device)


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""
    if model_name == 'WaveGlow':
        model_config = dict(
            n_mel_channels=args.n_mel_channels,
            n_flows=args.flows,
            n_group=args.groups,
            n_early_every=args.early_every,
            n_early_size=args.early_size,
            WN_config=dict(
                n_layers=args.wn_layers,
                kernel_size=args.wn_kernel_size,
                n_channels=args.wn_channels
            )
        )
        return model_config
    elif model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            # symbols
            n_symbols=len(get_symbols(args.symbol_set)),
            padding_idx=get_pad_idx(args.symbol_set),
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight,
            # energy predictor
            energy_predictor_kernel_size=args.energy_predictor_kernel_size,
            energy_predictor_filter_size=args.energy_predictor_filter_size,
            p_energy_predictor_dropout=args.p_energy_predictor_dropout,
            energy_predictor_n_layers=args.energy_predictor_n_layers,
            # energy conditioning
            energy_conditioning=args.energy_conditioning,
            energy_embedding_kernel_size=args.energy_embedding_kernel_size,
        )
        return model_config

    else:
        raise NotImplementedError(model_name)

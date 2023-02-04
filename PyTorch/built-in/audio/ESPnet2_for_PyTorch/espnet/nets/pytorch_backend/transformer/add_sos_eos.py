#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility functions for Transformer."""

import torch


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def npu_add_sos_eos(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """

    _eos = torch.ones_like(ys_pad) * eos
    ys_eos_pad = torch.where(ys_pad == ignore_id, _eos, ys_pad)
    _sos = torch.full((ys_pad.size(0), 1), sos, dtype=ys_pad.dtype, device=ys_pad.device)
    ys_in_pad = torch.cat((_sos, ys_eos_pad), dim=1)
    ys_in_pad = ys_in_pad[:, :-1]

    mask = ys_pad != ignore_id
    ys_eos_pos = mask.sum(dim=1).unsqueeze(-1)
    ys_out_pad = ys_pad.scatter(dim=1, index=ys_eos_pos, value=eos)
    return ys_in_pad, ys_out_pad


def npu_add_sos_eos_dyn(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """

    _eos = torch.ones_like(ys_pad) * eos
    ys_eos_pad = torch.where(ys_pad==ignore_id, _eos, ys_pad)
    _sos = torch.full((ys_pad.size(0), 1), sos, dtype=ys_pad.dtype, device=ys_pad.device)
    ys_in_pad = torch.cat((_sos, ys_eos_pad), dim=1)

    mask = ys_pad != ignore_id
    ys_eos_pos = mask.sum(dim=1).unsqueeze(-1)
    _ignore = torch.full((ys_pad.size(0), 1), ignore_id, dtype=ys_pad.dtype, device=ys_pad.device)
    ys_pad = torch.cat((ys_pad, _ignore), dim=1)
    ys_out_pad = ys_pad.scatter(dim=1, index=ys_eos_pos, value=eos)
    return ys_in_pad, ys_out_pad
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses
import logging
from collections import abc
from typing import Any

from detectron2.utils.registry import _convert_target_to_string, locate

__all__ = ["dump_dataclass", "instantiate"]


def dump_dataclass(obj: Any):
    """
    Dump a dataclass recursively into a dict that can be later instantiated.

    Args:
        obj: a dataclass object

    Returns:
        dict
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(
        obj, type
    ), "dump_dataclass() requires an instance of a dataclass."
    ret = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        ret[f.name] = v
    return ret


def instantiate(cfg):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """
    from omegaconf import ListConfig

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            return cls(**cfg)
        except TypeError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do

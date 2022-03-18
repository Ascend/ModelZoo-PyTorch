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

import os
import torch

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model

from densepose import add_densepose_config

_BASE_CONFIG_DIR = "configs"
_EVOLUTION_CONFIG_SUB_DIR = "evolution"
_HRNET_CONFIG_SUB_DIR = "HRNet"
_QUICK_SCHEDULES_CONFIG_SUB_DIR = "quick_schedules"
_BASE_CONFIG_FILE_PREFIX = "Base-"
_CONFIG_FILE_EXT = ".yaml"


def _get_base_config_dir():
    """
    Return the base directory for configurations
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", _BASE_CONFIG_DIR)


def _get_evolution_config_dir():
    """
    Return the base directory for evolution configurations
    """
    return os.path.join(_get_base_config_dir(), _EVOLUTION_CONFIG_SUB_DIR)


def _get_hrnet_config_dir():
    """
    Return the base directory for HRNet configurations
    """
    return os.path.join(_get_base_config_dir(), _HRNET_CONFIG_SUB_DIR)


def _get_quick_schedules_config_dir():
    """
    Return the base directory for quick schedules configurations
    """
    return os.path.join(_get_base_config_dir(), _QUICK_SCHEDULES_CONFIG_SUB_DIR)


def _collect_config_files(config_dir):
    """
    Collect all configuration files (i.e. densepose_*.yaml) directly in the specified directory
    """
    start = _get_base_config_dir()
    results = []
    for entry in os.listdir(config_dir):
        path = os.path.join(config_dir, entry)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(entry)
        if ext != _CONFIG_FILE_EXT:
            continue
        if entry.startswith(_BASE_CONFIG_FILE_PREFIX):
            continue
        config_file = os.path.relpath(path, start)
        results.append(config_file)
    return results


def get_config_files():
    """
    Get all the configuration files (relative to the base configuration directory)
    """
    return _collect_config_files(_get_base_config_dir())


def get_evolution_config_files():
    """
    Get all the evolution configuration files (relative to the base configuration directory)
    """
    return _collect_config_files(_get_evolution_config_dir())


def get_hrnet_config_files():
    """
    Get all the HRNet configuration files (relative to the base configuration directory)
    """
    return _collect_config_files(_get_hrnet_config_dir())


def get_quick_schedules_config_files():
    """
    Get all the quick schedules configuration files (relative to the base configuration directory)
    """
    return _collect_config_files(_get_quick_schedules_config_dir())


def get_model_config(config_file):
    """
    Load and return the configuration from the specified file (relative to the base configuration
    directory)
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    path = os.path.join(_get_base_config_dir(), config_file)
    cfg.merge_from_file(path)
    if not torch.cuda.is_available():
        cfg.MODEL_DEVICE = "cpu"
    return cfg


def get_model(config_file):
    """
    Get the model from the specified file (relative to the base configuration directory)
    """
    cfg = get_model_config(config_file)
    return build_model(cfg)


def setup(config_file):
    """
    Setup the configuration from the specified file (relative to the base configuration directory)
    """
    cfg = get_model_config(config_file)
    cfg.freeze()
    default_setup(cfg, {})

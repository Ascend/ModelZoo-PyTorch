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
""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_model_default_key', 'has_model_default_key', 'get_model_default_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_default_cfgs = dict()  # central repo for model default_cfgs


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module='', pretrained=False, exclude_filters=''):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
        pretrained (bool) - Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter

    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if module:
        models = list(_module_to_models[module])
    else:
        models = _model_entrypoints.keys()
    if filter:
        models = fnmatch.filter(models, filter)  # include these models
    if exclude_filters:
        if not isinstance(exclude_filters, list):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    return list(sorted(models, key=_natural_key))


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def list_modules():
    """ Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def has_model_default_key(model_name, cfg_key):
    """ Query model default_cfgs for existence of a specific key.
    """
    if model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]:
        return True
    return False


def is_model_default_key(model_name, cfg_key):
    """ Return truthy value for specified model default_cfg key, False if does not exist.
    """
    if model_name in _model_default_cfgs and _model_default_cfgs[model_name].get(cfg_key, False):
        return True
    return False


def get_model_default_value(model_name, cfg_key):
    """ Get a specific model default_cfg value by key. None if it doesn't exist.
    """
    if model_name in _model_default_cfgs:
        return _model_default_cfgs[model_name].get(cfg_key, None)
    else:
        return None


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained

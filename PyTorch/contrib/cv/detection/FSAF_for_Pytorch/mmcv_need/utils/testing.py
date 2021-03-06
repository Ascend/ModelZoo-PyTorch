# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

# Copyright (c) Open-MMLab.
from collections.abc import Iterable
from typing import Any, Dict, List


def _any(judge_result):
    """Since built-in ``any`` works only when the element of iterable is not
    iterable, implement the function."""
    if not isinstance(judge_result, Iterable):
        return judge_result

    try:
        for element in judge_result:
            if _any(element):
                return True
    except TypeError:
        # Maybe encouter the case: torch.tensor(True) | torch.tensor(False)
        if judge_result:
            return True
    return False


def assert_dict_contains_subset(dict_obj: Dict[Any, Any],
                                expected_subset: Dict[Any, Any]) -> bool:
    """Check if the dict_obj contains the expected_subset.

    Args:
        dict_obj (Dict[Any, Any]): Dict object to be checked.
        expected_subset (Dict[Any, Any]): Subset expected to be contained in
            dict_obj.

    Returns:
        bool: Whether the dict_obj contains the expected_subset.
    """

    for key, value in expected_subset.items():
        if key not in dict_obj.keys() or _any(dict_obj[key] != value):
            return False
    return True


def assert_attrs_equal(obj: Any, expected_attrs: Dict[str, Any]) -> bool:
    """Check if attribute of class object is correct.

    Args:
        obj (object): Class object to be checked.
        expected_attrs (Dict[str, Any]): Dict of the expected attrs.

    Returns:
        bool: Whether the attribute of class object is correct.
    """
    for attr, value in expected_attrs.items():
        if not hasattr(obj, attr) or _any(getattr(obj, attr) != value):
            return False
    return True


def assert_dict_has_keys(obj: Dict[str, Any],
                         expected_keys: List[str]) -> bool:
    """Check if the obj has all the expected_keys.

    Args:
        obj (Dict[str, Any]): Object to be checked.
        expected_keys (List[str]): Keys expected to contained in the keys of
            the obj.

    Returns:
        bool: Whether the obj has the expected keys.
    """
    return set(expected_keys).issubset(set(obj.keys()))


def assert_keys_equal(result_keys: List[str], target_keys: List[str]) -> bool:
    """Check if target_keys is equal to result_keys.

    Args:
        result_keys (List[str]): Result keys to be checked.
        target_keys (List[str]): Target keys to be checked.

    Returns:
        bool: Whether target_keys is equal to result_keys.
    """
    return set(result_keys) == set(target_keys)


def assert_is_norm_layer(module) -> bool:
    """Check if the module is a norm layer.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: Whether the module is a norm layer.
    """
    from .parrots_wrapper import _BatchNorm, _InstanceNorm
    from torch.nn import GroupNorm, LayerNorm
    norm_layer_candidates = (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm)
    return isinstance(module, norm_layer_candidates)


def assert_params_all_zeros(module) -> bool:
    """Check if the parameters of the module is all zeros.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: Whether the parameters of the module is all zeros.
    """
    weight_data = module.weight.data
    is_weight_zero = weight_data.allclose(
        weight_data.new_zeros(weight_data.size()))

    if hasattr(module, 'bias') and module.bias is not None:
        bias_data = module.bias.data
        is_bias_zero = bias_data.allclose(
            bias_data.new_zeros(bias_data.size()))
    else:
        is_bias_zero = True

    return is_weight_zero and is_bias_zero

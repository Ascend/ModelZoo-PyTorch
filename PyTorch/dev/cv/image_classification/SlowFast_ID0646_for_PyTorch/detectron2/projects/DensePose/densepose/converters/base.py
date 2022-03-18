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

from typing import Any, Tuple, Type
import torch


class BaseConverter:
    """
    Converter base class to be reused by various converters.
    Converter allows one to convert data from various source types to a particular
    destination type. Each source type needs to register its converter. The
    registration for each source type is valid for all descendants of that type.
    """

    @classmethod
    def register(cls, from_type: Type, converter: Any = None):
        """
        Registers a converter for the specified type.
        Can be used as a decorator (if converter is None), or called as a method.

        Args:
            from_type (type): type to register the converter for;
                all instances of this type will use the same converter
            converter (callable): converter to be registered for the given
                type; if None, this method is assumed to be a decorator for the converter
        """

        if converter is not None:
            cls._do_register(from_type, converter)

        def wrapper(converter: Any) -> Any:
            cls._do_register(from_type, converter)
            return converter

        return wrapper

    @classmethod
    def _do_register(cls, from_type: Type, converter: Any):
        cls.registry[from_type] = converter  # pyre-ignore[16]

    @classmethod
    def _lookup_converter(cls, from_type: Type) -> Any:
        """
        Perform recursive lookup for the given type
        to find registered converter. If a converter was found for some base
        class, it gets registered for this class to save on further lookups.

        Args:
            from_type: type for which to find a converter
        Return:
            callable or None - registered converter or None
                if no suitable entry was found in the registry
        """
        if from_type in cls.registry:  # pyre-ignore[16]
            return cls.registry[from_type]
        for base in from_type.__bases__:
            converter = cls._lookup_converter(base)
            if converter is not None:
                cls._do_register(from_type, converter)
                return converter
        return None

    @classmethod
    def convert(cls, instance: Any, *args, **kwargs):
        """
        Convert an instance to the destination type using some registered
        converter. Does recursive lookup for base classes, so there's no need
        for explicit registration for derived classes.

        Args:
            instance: source instance to convert to the destination type
        Return:
            An instance of the destination type obtained from the source instance
            Raises KeyError, if no suitable converter found
        """
        instance_type = type(instance)
        converter = cls._lookup_converter(instance_type)
        if converter is None:
            if cls.dst_type is None:  # pyre-ignore[16]
                output_type_str = "itself"
            else:
                output_type_str = cls.dst_type
            raise KeyError(f"Could not find converter from {instance_type} to {output_type_str}")
        return converter(instance, *args, **kwargs)


IntTupleBox = Tuple[int, int, int, int]


def make_int_box(box: torch.Tensor) -> IntTupleBox:
    int_box = [0, 0, 0, 0]
    int_box[0], int_box[1], int_box[2], int_box[3] = tuple(box.long().tolist())
    return int_box[0], int_box[1], int_box[2], int_box[3]

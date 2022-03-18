# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================from types import MethodType


def import_module_error_func(module_name):
    """When a function is imported incorrectly due to a missing module, raise
    an import error when the function is called."""

    def decorate(func):

        def new_func(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {func.__name__}.')

        return new_func

    return decorate


def import_module_error_class(module_name):
    """When a class is imported incorrectly due to a missing module, raise an
    import error when the class is instantiated."""

    def decorate(cls):

        def import_error_init(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {cls.__name__}.')

        cls.__init__ = MethodType(import_error_init, cls)
        return cls

    return decorate

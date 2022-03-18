# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================import functools
import warnings

import torch


class OutputHook:
    """Output feature map of some layers.

    Args:
        module (nn.Module): The whole module to get layers.
        outputs (tuple[str] | list[str]): Layer name to output. Default: None.
        as_tensor (bool): Determine to return a tensor or a numpy array.
            Default: False.
    """

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self.layer_outputs = {}
        self.handles = []
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                if not isinstance(output, torch.Tensor):
                    warnings.warn(f'Directly return the output from {name}, '
                                  f'since it is not a tensor')
                    self.layer_outputs[name] = output
                elif self.as_tensor:
                    self.layer_outputs[name] = output
                else:
                    self.layer_outputs[name] = output.detach().cpu().numpy()

            return hook

        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except AttributeError:
                    raise AttributeError(f'Module {name} not found')
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

# Copyright 2022 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
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
# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ort_custom_op_path = ''
try:
    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
except (ImportError, ModuleNotFoundError):
    warnings.warn('If input model has custom op from mmcv, \
        you may have to build mmcv with ONNXRuntime from source.')


class WrapFunction(nn.Module):
    """Wrap the function to be tested for torch.onnx.export tracking."""

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)


def ort_validate(model, feats, onnx_io='tmp.onnx'):
    """Validate the output of the onnxruntime backend is the same as the output
    generated by torch.

    Args:
        model (nn.Module | function): the function of model or model
            to be verified.
        feats (tuple(list(torch.Tensor)) | list(torch.Tensor) | torch.Tensor):
            the input of model.
        onnx_io (str): the name of onnx output file.
    """
    # if model is not an instance of nn.Module, then it is a normal
    # function and it should be wrapped.
    if isinstance(model, nn.Module):
        wrap_model = model
    else:
        wrap_model = WrapFunction(model)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            feats,
            onnx_io,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    if isinstance(feats, tuple):
        ort_feats = []
        for feat in feats:
            ort_feats += feat
    else:
        ort_feats = feats
    # default model name: tmp.onnx
    onnx_outputs = get_ort_model_output(ort_feats)

    # remove temp file
    if osp.exists(onnx_io):
        os.remove(onnx_io)

    if isinstance(feats, tuple):
        torch_outputs = convert_result_list(wrap_model.forward(*feats))
    else:
        torch_outputs = convert_result_list(wrap_model.forward(feats))
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


def get_ort_model_output(feat, onnx_io='tmp.onnx'):
    """Run the model in onnxruntime env.

    Args:
        feat (list[Tensor]): A list of tensors from torch.rand,
            each is a 4D-tensor.

    Returns:
        list[np.array]: onnxruntime infer result, each is a np.array
    """

    onnx_model = onnx.load(onnx_io)
    onnx.checker.check_model(onnx_model)

    session_options = ort.SessionOptions()
    # register custom op for onnxruntime
    if osp.exists(ort_custom_op_path):
        session_options.register_custom_ops_library(ort_custom_op_path)
    sess = ort.InferenceSession(onnx_io, session_options)
    if isinstance(feat, torch.Tensor):
        onnx_outputs = sess.run(None,
                                {sess.get_inputs()[0].name: feat.numpy()})
    else:
        onnx_outputs = sess.run(None, {
            sess.get_inputs()[i].name: feat[i].numpy()
            for i in range(len(feat))
        })
    return onnx_outputs


def convert_result_list(outputs):
    """Convert the torch forward outputs containing tuple or list to a list
    only containing torch.Tensor.

    Args:
        output (list(Tensor) | tuple(list(Tensor) | ...): the outputs
        in torch env, maybe containing nested structures such as list
        or tuple.

    Returns:
        list(Tensor): a list only containing torch.Tensor
    """
    # recursive end condition
    if isinstance(outputs, torch.Tensor):
        return [outputs]

    ret = []
    for sub in outputs:
        ret += convert_result_list(sub)
    return ret

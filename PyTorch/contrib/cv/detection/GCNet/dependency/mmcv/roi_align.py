import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import deprecated_api_warning, ext_loader
import traceback

ext_module = ext_loader.load_ext('_ext',
                                 ['roi_align_forward', 'roi_align_backward'])

def set_device(obj, device='cpu'):
    if isinstance(obj, (tuple, list)):
        dump = []
        for item in obj:
            dump.append(set_device(item, device))
        return dump
    elif isinstance(obj, dict):
        dump = {}
        for k, v in obj.items():
            dump[k] = set_device(v, device)
        return dump
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def dump_tensor(output, name):
    dump = set_device(output, 'cpu')
    torch.save(dump, name)
    print('%s dump success!' % (name))


def load_tensor(name, device):
    output = torch.load(name)
    dump = set_device(output, device)
    print('%s load success!' % (name), ' dtype:',dump.dtype, ' size:',dump.size())
    return dump

class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from ..onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        if has_custom_op:
            return g.op(
                'mmcv::MMCVRoiAlign',
                input,
                rois,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode,
                aligned_i=aligned)
        else:
            from torch.onnx.symbolic_opset9 import sub, squeeze
            from torch.onnx.symbolic_helper import _slice_helper
            from torch.onnx import TensorProtoDataType
            # batch_indices = rois[:, 0].long()
            batch_indices = _slice_helper(
                g, rois, axes=[1], starts=[0], ends=[1])
            batch_indices = squeeze(g, batch_indices, 1)
            batch_indices = g.op(
                'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
            # rois = rois[:, 1:]
            rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
            if aligned:
                # rois -= 0.5/spatial_scale
                aligned_offset = g.op(
                    'Constant',
                    value_t=torch.tensor([0.5 / spatial_scale],
                                         dtype=torch.float32))
                rois = sub(g, rois, aligned_offset)
            # roi align
            return g.op(
                'RoiAlign',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=max(0, sampling_ratio),
                mode_s=pool_mode)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        # print("forward attr type:")
        # print("input:",input.dtype)
        # print("rois:",rois.dtype)
        # print("output:",output.dtype)
        # print("argmax_y:",argmax_y.dtype)
        # print("argmax_x:",argmax_x.dtype)
        # rois = rois.half()
        # input = input.half()
        # output = output.half()
        # argmax_y = argmax_y.half()
        # argmax_x = argmax_x.half()    
        # print('half change')
        
        # ext_module.roi_align_forward(
        #     input,
        #     rois.half(),
        #     output,
        #     argmax_y,
        #     argmax_x,
        #     aligned_height=ctx.output_size[0],
        #     aligned_width=ctx.output_size[1],
        #     spatial_scale=ctx.spatial_scale,
        #     sampling_ratio=ctx.sampling_ratio,
        #     pool_mode=ctx.pool_mode,
        #     aligned=ctx.aligned)
#         print('================roi op')
#         print('rois.size()',rois.size())
#         print('input.size()', input.size())
#         print('ctx.output_size', ctx.output_size)
        # ctx.spatial_scale = 0.25
        # ctx.sampling_ratio = 0

        roi_end_mode = 2
#         dump_tensor(input,"input.pt")
#         dump_tensor(rois,"rois.pt")
#         print(torch.npu.synchronize(),"roi_align")
#         print("ctx.spatial_scale:",ctx.spatial_scale)
#         print("ctx.output_size[0]:",ctx.output_size[0])
#         print("ctx.output_size[1]:",ctx.output_size[1])
#         print("ctx.sampling_ratio:",ctx.sampling_ratio)
#         print("roi_end_mode:",roi_end_mode)

        
        
        output = torch_npu.npu_roi_align(
            input,rois,ctx.spatial_scale,
            ctx.output_size[0],ctx.output_size[1],
            ctx.sampling_ratio,roi_end_mode)

        # print('fwd finish')
        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output
        # return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        # ext_module.roi_align_backward(
        #     grad_output,
        #     rois,
        #     argmax_y,
        #     argmax_x,
        #     grad_input,
        #     aligned_height=ctx.output_size[0],
        #     aligned_width=ctx.output_size[1],
        #     spatial_scale=ctx.spatial_scale,
        #     sampling_ratio=ctx.sampling_ratio,
        #     pool_mode=ctx.pool_mode,
        #     aligned=ctx.aligned)
        # ctx.spatial_scale = 0.25
        # ctx.sampling_ratio = 0
        # print(torch.npu.synchronize(),"def backward(ctx, grad_output):")
        # print("roi_end_mode:",roi_end_mode)
        roi_end_mode = 2
#         dump_tensor(grad_output,"grad_output.pt")
#         dump_tensor(rois,"rois.pt")
#         print(torch.npu.synchronize(),"roi_align")
#         print("ctx.input_shape:",ctx.input_shape)
#         print("ctx.output_size[0]:",ctx.output_size[0])
#         print("ctx.output_size[1]:",ctx.output_size[1])
#         print("ctx.spatial_scale:",ctx.spatial_scale)
#         print("ctx.sampling_ratio:",ctx.sampling_ratio)
#         print("roi_end_mode:",roi_end_mode)

        
        grad_input = torch_npu.npu_roi_alignbk(
            grad_output,rois,ctx.input_shape,
            ctx.output_size[0],ctx.output_size[1],
            ctx.spatial_scale,ctx.sampling_ratio, roi_end_mode)
        
        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlign')
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.use_torchvision = use_torchvision

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
#         print('roi stack trace:')
        #traceback.print_stack()
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            if 'aligned' in tv_roi_align.__code__.co_varnames:
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.aligned)
            else:
                if self.aligned:
                    rois -= rois.new_tensor([0.] +
                                            [0.5 / self.spatial_scale] * 4)
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(input.float(), rois.float(), self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        s += f'use_torchvision={self.use_torchvision})'
        return s


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
#
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
# import torch
# from torch import nn

# from torch.nn.modules.utils import _pair
# from torch.autograd import Function
# from torch.autograd.function import once_differentiable

# class _ROIAlign(Function):
#     @staticmethod
#     def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
#         ctx.save_for_backward(roi)
#         ctx.output_size = _pair(output_size)
#         ctx.spatial_scale = spatial_scale
#         ctx.sampling_ratio = sampling_ratio
#         ctx.input_shape = input.size()
#         ctx.aligned = aligned
#         roi_end_mode = 0
#         output = torch_npu.npu_roi_align(
#             input, roi, spatial_scale,
#             output_size[0], output_size[1], sampling_ratio, roi_end_mode)

#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         (rois,) = ctx.saved_tensors
#         output_size = ctx.output_size
#         spatial_scale = ctx.spatial_scale
#         sampling_ratio = ctx.sampling_ratio
#         bs, ch, h, w = ctx.input_shape

#         grad_input = torch_npu.npu_roi_alignbk(
#             grad_output, rois, ctx.input_shape,
#             output_size[0], output_size[1],
#             spatial_scale, sampling_ratio)

#         return grad_input, None, None, None, None, None

# roi_align = _ROIAlign.apply

# # NOTE: torchvision's RoIAlign has a different default aligned=False
# class RoIAlign(nn.Module):
#     def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
#         """
#         Args:
#             output_size (tuple): h, w
#             spatial_scale (float): scale the input boxes by this number
#             sampling_ratio (int): number of inputs samples to take for each output
#                 sample. 0 to take samples densely.
#             aligned (bool): if False, use the legacy implementation in
#                 Detectron. If True, align the results more perfectly.

#         Note:
#             The meaning of aligned=True:

#             Given a continuous coordinate c, its two neighboring pixel indices (in our
#             pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
#             c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
#             from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
#             roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
#             pixel indices and therefore it uses pixels with a slightly incorrect alignment
#             (relative to our pixel model) when performing bilinear interpolation.

#             With `aligned=True`,
#             we first appropriately scale the ROI and then shift it by -0.5
#             prior to calling roi_align. This produces the correct neighbors; see
#             detectron2/tests/test_roi_align.py for verification.

#             The difference does not make a difference to the model's performance if
#             ROIAlign is used together with conv layers.
#         """
#         super(RoIAlign, self).__init__()
#         self.output_size = output_size
#         self.spatial_scale = spatial_scale
#         self.sampling_ratio = sampling_ratio
#         self.aligned = aligned

#     def forward(self, input, rois):
#         """
#         Args:
#             input: NCHW images
#             rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
#         """
#         assert rois.dim() == 2 and rois.size(1) == 5
#         return roi_align(
#              input.float(), rois, self.output_size,
#              self.spatial_scale, self.sampling_ratio, self.aligned
#         )

#     def __repr__(self):
#         tmpstr = self.__class__.__name__ + "("
#         tmpstr += "output_size=" + str(self.output_size)
#         tmpstr += ", spatial_scale=" + str(self.spatial_scale)
#         tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
#         tmpstr += ", aligned=" + str(self.aligned)
#         tmpstr += ")"
#         return tmpstr


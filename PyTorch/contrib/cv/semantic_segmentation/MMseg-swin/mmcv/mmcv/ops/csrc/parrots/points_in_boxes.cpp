# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void points_in_boxes_part_forward_impl(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points) {
  DISPATCH_DEVICE_IMPL(points_in_boxes_part_forward_impl, batch_size, boxes_num,
                       pts_num, boxes, pts, box_idx_of_points);
}

void points_in_boxes_all_forward_impl(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points) {
  DISPATCH_DEVICE_IMPL(points_in_boxes_all_forward_impl, batch_size, boxes_num,
                       pts_num, boxes, pts, box_idx_of_points);
}

void points_in_boxes_part_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                  Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box params pts: (B, npoints, 3)
  // [x, y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints),
  // default -1
  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);
  points_in_boxes_part_forward_impl(batch_size, boxes_num, pts_num,
                                    boxes_tensor, pts_tensor,
                                    box_idx_of_points_tensor);
}

void points_in_boxes_all_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center. params pts: (B, npoints, 3) [x, y, z]
  // in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default -1
  int batch_size = boxes_tensor.size(0);
  int boxes_num = boxes_tensor.size(1);
  int pts_num = pts_tensor.size(1);
  points_in_boxes_all_forward_impl(batch_size, boxes_num, pts_num, boxes_tensor,
                                   pts_tensor, box_idx_of_points_tensor);
}

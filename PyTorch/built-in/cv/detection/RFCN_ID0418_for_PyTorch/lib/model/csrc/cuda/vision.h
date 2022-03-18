// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <THC/THC.h>
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);


std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);

int PSROIPoolForwardLauncher(at::Tensor bottom_data,
                             const float spatial_scale,
                             const int num_rois,
                             const int height,
                             const int width,
                             const int channels,
                             const int pooled_height,
                             const int pooled_width,
                             at::Tensor bottom_rois,
                             const int group_size,
                             const int output_dim,
                             at::Tensor top_data,
                             at::Tensor mapping_channel,
                             cudaStream_t stream);


int PSROIPoolBackwardLauncher(at::Tensor top_diff,
                              at::Tensor mapping_channel,
                              const int batch_size,
                              const int num_rois,
                              const float spatial_scale,
                              const int channels,
                              const int height,
                              const int width,
                              const int pooled_width,
                              const int pooled_height,
                              const int output_dim,
                              at::Tensor bottom_diff,
                              at::Tensor bottom_rois,
                              cudaStream_t stream);


int PSROIAlignForwardLaucher(at::Tensor bottom_data,
                             at::Tensor bottom_rois,
                             at::Tensor top_data,
                             at::Tensor argmax_data,
                             float spatial_scale,
                             int group_size,
                             int sampling_ratio,
                             cudaStream_t stream);


int PSROIAlignBackwardLaucher(at::Tensor top_diff,
                              at::Tensor argmax_data,
                              at::Tensor bottom_rois,
                              at::Tensor bottom_diff,
                              float spatial_scale,
                              int group_size,
                              int sampling_ratio,
                              cudaStream_t stream);

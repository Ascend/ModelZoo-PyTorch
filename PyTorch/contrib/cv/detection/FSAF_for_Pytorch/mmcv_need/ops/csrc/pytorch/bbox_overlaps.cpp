// Copyright 2021 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void BBoxOverlapsCUDAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset);

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  BBoxOverlapsCUDAKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}
#endif

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
  if (bboxes1.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(bboxes1);
    CHECK_CUDA_INPUT(bboxes2);
    CHECK_CUDA_INPUT(ious);

    bbox_overlaps_cuda(bboxes1, bboxes2, ious, mode, aligned, offset);
#else
    AT_ERROR("bbox_overlaps is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("bbox_overlaps is not implemented on CPU");
  }
}

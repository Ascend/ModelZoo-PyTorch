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

#include "onnxruntime_register.h"

#include "nms.h"
#include "ort_mmcv_utils.h"
#include "roi_align.h"
#include "soft_nms.h"

const char *c_MMCVOpDomain = "mmcv";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
MMCVRoiAlignCustomOp c_MMCVRoiAlignCustomOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_NmsOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoiAlignCustomOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

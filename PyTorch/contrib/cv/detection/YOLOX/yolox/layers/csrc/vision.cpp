// Copyright 2021 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "cocoeval/cocoeval.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
    m.def(
        "COCOevalEvaluateImages",
        &COCOeval::EvaluateImages,
        "COCOeval::EvaluateImages");
    pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
        .def(pybind11::init<uint64_t, double, double, bool, bool>());
    pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation")
        .def(pybind11::init<>());
}

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN pip install torch==1.3.1

WORKDIR /code
ADD . .

# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
FROM nvidia/cuda:10.0-base
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}
#FROM nvidia/cuda:10.0-base

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*
RUN conda install -y python=3.7.5
#RUN pip install torch==1.5.0  torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.5.0 torchvision==0.6.0
WORKDIR opt

# Install Python dependencies
WORKDIR /workspace/single_stage_detector

COPY requirements.txt .
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir -r requirements.txt

# Copy SSD code and build
COPY . .
RUN pip install .

# Configure environment variables
ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1
ENV TORCH_HOME=/workspace/single_stage_detector/torch-model-cache

#!/usr/bin/env bash
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download checkpoint
if [ -f "${CHECKPOINT_DIR}/nvidia_fastpitch_200518.pt" ]; then
  echo "Checkpoint already downloaded."
elif [ -f "${WORKDIR}/pretrained_models/fastpitch/nvidia_fastpitch_200518.pt" ]; then
  echo "Linking existing checkpoint..."
  ln -s "${WORKDIR}/pretrained_models/fastpitch/nvidia_fastpitch_200518.pt" "${CHECKPOINT_DIR}/nvidia_fastpitch_200518.pt"
elif [ -f "${PWD}/pretrained_models/fastpitch/nvidia_fastpitch_200518.pt" ]; then
  echo "Linking existing checkpoint..."
  ln -s "${PWD}/pretrained_models/fastpitch/nvidia_fastpitch_200518.pt" "${CHECKPOINT_DIR}/nvidia_fastpitch_200518.pt"
else
  echo "Downloading checkpoint ..."
  fastpitch_20_02=`sed '/^fastpitch_20_02=/!d;s/.*=//' url.ini`
  wget --content-disposition ${fastpitch_20_02} -O \
    fastpitch_pyt_amp_ckpt_v1_20.02.0.zip || {
    echo "ERROR: Failed to download checkpoint from NGC"
    exit 1
  }
  unzip fastpitch_pyt_amp_ckpt_v1_20.02.0.zip -d ${CHECKPOINT_DIR}
  rm fastpitch_pyt_amp_ckpt_v1_20.02.0.zip
  echo "ok"
fi

# Download dataset
if [ -d "${DATASETS_DIR}/LJSpeech-1.1" ]; then
  echo "Dataset already downloaded."
elif [ -d "${WORKDIR}/LJSpeech-1.1" ]; then
  echo "Linking existing dataset from ${WORKDIR}/LJSpeech-1.1"
  mkdir -p "${DATASETS_DIR}/LJSpeech-1.1"
  ln -s "${WORKDIR}/LJSpeech-1.1" "${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch"
elif [ -d "${PWD}/LJSpeech-1.1" ]; then
  echo "Linking existing dataset from ${PWD}/LJSpeech-1.1"
  mkdir -p "${DATASETS_DIR}/LJSpeech-1.1"
  ln -s "${PWD}/LJSpeech-1.1" "${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch"
else
  echo "Downloading dataset ..."
  LJSpeech=`sed '/^LJSpeech=/!d;s/.*=//' url.ini`
  wget ${LJSpeech} ||
    {
      echo "ERROR: Failed to download dataset from NGC"
      exit 1
    }
  mkdir -p "${DATASETS_DIR}/LJSpeech-1.1"
  tar -jxf LJSpeech-1.1.tar.bz2 --directory "${DATASETS_DIR}/LJSpeech-1.1"
  mv "${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1" "${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch"
  rm LJSpeech-1.1.tar.bz2
  echo "ok"
fi

echo "Downloading cmudict-0.7b ..."
cmudict=`sed '/^cmudict=/!d;s/.*=//' url.ini`
wget ${cmudict} -qO cmudict/cmudict-0.7b

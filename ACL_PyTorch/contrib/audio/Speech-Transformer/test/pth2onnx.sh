# Copyright 2021 Huawei Technologies Co., Ltd
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
source path.sh
# convert to onnx
python3.7 SpeechTransformer_pth2onnx_encoder.py --pth-path './final.pth.tar' --encoder-path './encoder.onnx'
python3.7 SpeechTransformer_pth2onnx_decoder.py --pth-path './final.pth.tar' --decoder-path './decoder.onnx'
python3.7 SpeechTransformer_pth2onnx_tgt_word_prj.py --pth-path './final.pth.tar' --tgt-word-prj-path './tgt_word_prj.onnx'
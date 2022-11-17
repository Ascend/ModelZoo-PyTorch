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

from transformers import Wav2Vec2Processor, Data2VecAudioForCTC
import torch
import torch.onnx
processor = Wav2Vec2Processor.from_pretrained("data2vec_pytorch_model")
model = Data2VecAudioForCTC.from_pretrained("data2vec_pytorch_model")
model.eval()
input_size = 559280 #The max length of the audio file
AUDIO_MAXLEN = input_size
dummy_input = torch.randn(1, input_size, requires_grad=True)
torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "data2vec.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=11,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput': {0 : 'batch_size'},    # variable length axes
                        'modelOutput': {0 : 'batch_size'}})

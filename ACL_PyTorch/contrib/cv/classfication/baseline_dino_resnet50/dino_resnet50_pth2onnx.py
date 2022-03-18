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
import torch
import torchvision.models as models
import torch.nn as nn


model=models.resnet50(pretrained=True)
embed_dim = model.fc.weight.shape[1]
model.fc = nn.Identity()

state_dict = torch.load('dino_resnet50_pretrain.pth', map_location='cpu')
temp=nn.Linear(2048,1000)
state_dict2 = torch.load('dino_resnet50_linearweights.pth', map_location='cpu')["state_dict"]
temp.weight.data = state_dict2['module.linear.weight']
temp.bias.data = state_dict2['module.linear.bias']
model.load_state_dict(state_dict, strict=True)

model.fc=temp
model.eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)
model.to(device='cpu')
torch_out = model(x.to(device="cpu"))

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "dino_resnet50.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
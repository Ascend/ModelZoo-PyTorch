# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import sys

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import torch_aie
from torch_aie import _enums


def accuracy(output, target, topk):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model):
    model.eval()
    avg_top1, avg_top5 = 0, 0
    top1, top5 = 1, 5
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            output = model(images)
            acc = accuracy(output, target, topk=(top1, top5))
            avg_top1 += acc[0]
            avg_top5 += acc[1]
            print("top1 is {0}， top5 is {1}, step is {2}".format(avg_top1/i, avg_top5/i, i))


def inference(model_path, data_path):
    torch_aie.set_device(0)
    batchsize = 64
    model = models.__dict__["resnet50"]()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    input_data = torch.ones(batchsize, 3, 224, 224)
    trace_model = torch.jit.trace(model, input_data)
    input_info = [torch_aie.Input((batchsize, 3, 224, 224))]
    pt_model = torch_aie.compile(
        trace_model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP32,
        soc_version="Ascend310P3")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        drop_last = True)
    validate(val_loader, pt_model)


def trace_ts_model(model_path, save_path="./resnet50.ts"):
    model = models.__dict__["resnet50"]()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    input_data = torch.ones(64, 3, 224, 224)
    ts_model = torch.jit.trace(model, input_data)
    ts_model.save(save_path)
    print("trace resnet50 done. save path is ", save_path)


def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    trace_ts_model(model_path)
    inference(model_path, data_path)

if __name__ == '__main__':
    main()

# Copyright 2023 Huawei Technologies Co., Ltd
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

import re
import sys
import time
import argparse

import torch
import torch_aie
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_args_parser():
    args = argparse.ArgumentParser(add_help=False)
    args.add_argument('--batch_size', default=64, type=int)
    args.add_argument("--device_id", default=0, type=int)
    args.add_argument("--warmup_loop", default=5, type=int)
    args.add_argument("--pre_trained", default="/path/to/densenet121-a639ec97.pth", type=str)
    args.add_argument("--dataset", default="/path/to/imagenet/val", type=str)
    return args


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


def validate(val_loader, model, args):
    model.eval()
    inference_time = []
    top1_rets, top5_rets = [], []
    top1, top5 = 1, 5
    stream = torch_aie.npu.Stream("npu:" + str(args.device_id))
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to("npu:" + str(args.device_id))
            with torch_aie.npu.stream(stream):
                ts = time.time()
                output = model(images)
                stream.synchronize()
                te = time.time()
                inference_time.append(te - ts)
                output = output.to("cpu")
                acc = accuracy(output, target, topk=(top1, top5))
                top1_rets.append(acc[0])
                top5_rets.append(acc[1])
    print("top1 is {0}, top5 is {1}, QPS is {2}, batch_size is {3}".
        format(sum(top1_rets)/len(top1_rets), sum(top5_rets)/len(top5_rets),
        (len(inference_time) - args.warmup_loop) /sum(inference_time[args.warmup_loop:-1])*args.batch_size,
        args.batch_size))


def get_inputs(batch_list):
    inputs = []
    for batch in batch_list:
        inputs.append([torch_aie.Input((batch, 3, 224, 224))])
    return inputs


def inference(args):
    torch_aie.set_device(args.device_id)
    model = torchvision.models.densenet121()
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
    )
    state_dict = torch.load(args.pre_trained, map_location='cpu')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()

    input_data = torch.ones(args.batch_size, 3, 224, 224)
    trace_model = torch.jit.trace(model, input_data)
    input_info = get_inputs([args.batch_size])
    pt_model = torch_aie.compile(
        trace_model,
        inputs=input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        soc_version="Ascend310P3")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        args.dataset,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last = True)
    validate(val_loader, pt_model, args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    inference(args)
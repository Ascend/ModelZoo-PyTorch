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

import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm

import torch_aie
from torch_aie import _enums


def compute_acc(y_pred, y_true, topk_list=(1, 5)):
    maxk = max(topk_list)
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res = []
    for k in topk_list:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, args):
    img_resize = args.model_config[args.image_size]['resize']
    img_centercrop = args.model_config[args.image_size]['centercrop']
    mean, std = args.model_config[args.image_size]['mean'], args.model_config[args.image_size]['std']

    val_dataset = datasets.ImageFolder(
        args.data_path,
        transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_centercrop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True)

    avg_top1, avg_top5 = 0, 0
    top1, top5 = 1, 5
    print('==================== Start Validation ====================')
    for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        pred = model(images).cpu()
        acc = compute_acc(pred, target, topk_list=(top1, top5))
        avg_top1 += acc[0].item()
        avg_top5 += acc[1].item()

        step = i + 1
        if step % 100 == 0:
            print(f'top1 is {avg_top1 / step}, top5 is {avg_top5 / step}, step is {step}')


def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer Evaluation.')
    parser.add_argument('--data_path', type=str, help='Evaluation dataset path')
    parser.add_argument('--model_path', type=str, default='./inceptionv3.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=299, help='Image size')
    return parser.parse_args()

def main():
    args = parse_args()
    args.model_config = {
        299: {
            'resize': 299,
            'centercrop': 299,
            'mean': [0.485,0.456,0.406],
            'std': [0.299,0.224,0.225],
        }
    }

    torch_aie.set_device(0)

    model = torch.jit.load(args.model_path)
    model.eval()

    input_info = [torch_aie.Input((args.batch_size, 3, args.image_size, args.image_size))]
    print('Start compiling model.')
    compiled_model = torch_aie.compile(
        model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version="Ascend310P3")
    print('Model compiled successfully.')

    validate(compiled_model, args)


if __name__ == '__main__':
    main()

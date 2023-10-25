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
import torch_aie
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_aie import _enums
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='HRNet Evaluation.')
    parser.add_argument('--data_path', type=str, help='Evaluation dataset path')
    parser.add_argument('--model_path', type=str, default='./hrnet.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    return parser.parse_args()


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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_path, transforms.Compose([
            transforms.Resize(int(args.image_size / 0.875)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    avg_top1, avg_top5 = 0, 0
    top1, top5 = 1, 5
    print('==================== Start Validation ====================')
    for i, (images, target) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        pred = model(images)
        acc = compute_acc(pred, target, topk_list=(top1, top5))
        avg_top1 += acc[0].item()
        avg_top5 += acc[1].item()

        step = i + 1
        if step % 100 == 0:
            print(f'top1 is {avg_top1 / step}, top5 is {avg_top5 / step}, step is {step}')


if __name__ == '__main__':
    args = parse_args()
    ts_model = torch.jit.load(args.model_path)
    min_shape = (1, 3, 224, 224)
    max_shape = (32, 3, 224, 224)
    input_info = [torch_aie.Input(min_shape=(1, 3, 224, 224), max_shape=(32, 3, 224, 224))]
    torchaie_model = torch_aie.compile(
        ts_model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP32,
        soc_version='Ascend310P3'
    )
    torchaie_model.eval()
    validate(torchaie_model, args)

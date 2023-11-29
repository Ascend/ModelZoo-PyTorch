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
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import torch_aie
from torch_aie import _enums


class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(self.root_dir)))}
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(img)
        if self.transform:
            pilimg = self.transform(pilimg)

        # Get the target (label) from the directory name
        target_name = os.path.basename(os.path.dirname(img_path))
        target_idx = self.class_to_idx[target_name]
        target_tensor = torch.tensor(target_idx)

        return pilimg, target_tensor


def validate(model, args):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    val_dataset = CustomImageFolder(
        args.data_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True)

    avg_top1, avg_top5 = 0, 0
    top1, top5 = 1, 5
    step = 0
    print('==================== Start Validation ====================')
    for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images_npu = images.to("npu:" + str(args.device_id))
        pred = model(images_npu).cpu()
        acc = compute_acc(pred, target, topk_list=(top1, top5))
        avg_top1 += acc[0].item()
        avg_top5 += acc[1].item()
        step = i + 1

    print(f'top1 is {avg_top1 / step}, top5 is {avg_top5 / step}, step is {step}')


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


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNet_V2 Evaluation.')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    parser.add_argument('--data_path', type=str, help='Evaluation dataset path')
    parser.add_argument('--model_path', type=str, default='./mobilenet_v2.ts',
                        help='Original TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    return parser.parse_args()


def main():
    args = parse_args()

    torch_aie.set_device(args.device_id)

    model = torch.jit.load(args.model_path)
    model.eval()

    input_info = [torch_aie.Input((args.batch_size, 3, args.image_size, args.image_size))]
    print('Start compiling model.')
    compiled_model = torch_aie.compile(
        model,
        inputs=input_info,
        precision_policy=_enums.PrecisionPolicy.FP16,
        soc_version="Ascend310P3")
    print('Model compiled successfully.')

    validate(compiled_model, args)


if __name__ == '__main__':
    main()

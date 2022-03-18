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
import os
import sys
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # output folder
    output_dir = os.path.join(root_path,'runs/demo_result', 'vis_result_{}_{}_{}_{}'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().npu()
    model.eval()

    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img)]
    else:
        img_paths = [args.input_img]
    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        images = transform(image).unsqueeze(0).npu()
        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, cfg.DATASET.NAME)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        mask.save(os.path.join(output_dir, outname))


if __name__ == '__main__':
    demo()

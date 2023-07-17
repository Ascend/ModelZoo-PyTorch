
# Copyright 2023 The HuggingFace Datasets Authors and the current dataset script contributor.
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
from argparse import ArgumentParser

import mmcv

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(CURRENT_PATH, 'url.ini'), 'r') as _f:
    _content = _f.read()
    test_image_url = _content.split('test_image_url=')[1].split('\n')[0]


def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = test_image_url
    urlretrieve(IMAGE_URL, 'tmp.jpg')


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='2009_002112.jpg', help='Image file')
    parser.add_argument('--config', default='./configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py', help='Config file')
    parser.add_argument('--checkpoint', default='output/FCN/0,1,2,3,4,5,6,7/ckpt/latest.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    # if hasattr(model, 'module'):
    #     model = model.module
    # model.show_result(args.img, result, palette=args.palette, out_file="result_"+args.img,show=False)
    prediction = show_result_pyplot(model, args.img, result, get_palette(args.palette))
    mmcv.imwrite(prediction, "result_"+args.img)



if __name__ == '__main__':
    main()
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

import os.path as osp
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="GFocalV2-master/demo/demo.jpg", help='Image file')
    parser.add_argument('--config', default="GFocalV2-master/configs/gfocal/gfocal_r50_fpn_1x.py", help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='GFocalV2-master/work_dirs/gfocal_r50_fpn_1x/latest.pth')
    parser.add_argument(
        '--device', default='npu:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='.', help='directory where painted images will be saved')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    if args.show_dir:
        out_file = osp.join(args.show_dir, osp.basename(args.img))
    else:
        out_file = None
    model.show_result(
                    args.img,
                    result,
                    show=args.show,
                    out_file=out_file,
                    score_thr=args.score_thr)


if __name__ == '__main__':
    main()
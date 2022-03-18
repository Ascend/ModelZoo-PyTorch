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


import argparse
import sys
import torch
import cv2

from vision.utils.misc import Timer
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite

parser = argparse.ArgumentParser()
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", default="models/1p/mb2-ssd-lite-Epoch-0-Loss-12.09216200136671.pth", type=str)
parser.add_argument('--img', default="demo.jpg", help="image file")
parser.add_argument("--label_file", default="models/1p/voc-model-labels.txt", type=str, help="The label file path.")
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
timer = Timer()
if __name__ == '__main__':
    args = parser.parse_args()
    if args.device == 'npu':
        args.device = 'npu:{}'.format(args.gpu)
        torch.npu.set_device(args.device)
    elif args.device == 'gpu':
        args.device = 'cuda:{}'.format(args.gpu)
        torch.backends.cudnn.benchmark = True

    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        create_predictor = lambda net: create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=args.device)
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        create_predictor = lambda net: create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method,
                                                                        device=args.device)
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        create_predictor = lambda net: create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        create_predictor = lambda net: create_squeezenet_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                            device=args.device)
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult, device=args.device)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        create_predictor = lambda net: create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method,
                                                                             device=args.device)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # create model
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    net = create_net(len(class_names))
    timer.start("Load Model")
    pretrained_dic = torch.load(args.trained_model, map_location='cpu')['state_dict']
    pretrained_dic = {k.replace('module.', ''): v for k, v in pretrained_dic.items()}
    net.load_state_dict(pretrained_dic)

    net = net.to(args.device)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    # create predictor
    predictor = create_predictor(net)

    # load imge
    image = cv2.imread(args.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image)
    print('\n')
    print('boxes: ', boxes)
    print('lables: ', labels)
    print('probs: ', probs)

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
import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from yolox.utils.boxes import postprocess

from yolox.utils.demo_utils import demo_postprocess
sys.path.append('./YOLOX')


def get_output_data(dump_dir, idx, dtype=np.float32):
    shapes = [[-1, 4, 80, 80], [-1, 1, 80, 80], [-1, 80, 80, 80],
              [-1, 4, 40, 40], [-1, 1, 40, 40], [-1, 80, 40, 40],
              [-1, 4, 20, 20], [-1, 1, 20, 20], [-1, 80, 20, 20]]
    res = []
    for index, shape in enumerate(shapes):
        file_name = os.path.join(dump_dir, f"{idx:0>12d}_{index}.bin")
        data = np.formfile(file_name, dtype=dtype).reshape(shape)
        res.append(torch.from_numpy(data))

    return res


def main():
    parser = argparse.ArgumentParser(description='YOLOX Postprocess')
    parser.add_argument('--dataroot', dest='dataroot',
                        help='data root dirname', default='/opt/npu/coco',
                        type=str)
    parser.add_argument('--dump_dir', dest='dump_dir',
                        help='dump dir for bin files', default='./result/dumpOutput_device0/',
                        type=str)

    parser.add_argument('--batch', dest='batch',
                        help='batch for dataloader', default=1, type=int)
    opt = parser.parse_args()

    if os.path.exists(opt.dump_dir):
        os.system("rm-rf " + opt.dump_dir)
    else:
        os.system("mkdir " + opt.dump_dir)

    valdataset = COCODataset(
        data_dir=opt.dataroot,
        json_file='instances_val2017.json',
        name="val2017",
        img_size=(640, 640),
        preproc=ValTransform(legacy=False),
    )
    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {"num_workers": 8, "pin_memory": True,
                         "sampler": sampler, "batch_size": opt.batch}

    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    data_list = []
    coco_evaluator = COCOEvaluator(val_loader, img_size=(
        640, 640), confthre=0.001, nmsthre=0.65, num_classes=80)

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(tqdm(val_loader)):

        opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9 = get_output_data(
            opt.dump_dir, cur_iter)
        opt2 = opt2.sigmoid()
        opt3 = opt3.sigmoid()

        opt5 = opt5.sigmoid()
        opt6 = opt6.sigmoid()

        opt8 = opt8.sigmoid()
        opt9 = opt9.sigmoid()
        output1 = torch.cat((opt1, opt2, opt3), dim=1)
        output2 = torch.cat((opt4, opt5, opt6), dim=1)
        output3 = torch.cat((opt7, opt8, opt9), dim=1)

        output1 = output1.view(1, 85, -1)
        output2 = output2.view(1, 85, -1)
        output3 = output3.view(1, 85, -1)

        outputs = torch.cat((output1, output2, output3), dim=2)
        outputs = outputs.transpose(2, 1)

        outputs = demo_postprocess(outputs, [640, 640])

        outputs = postprocess(outputs, num_classes=80,
                              conf_thre=0.001, nms_thre=0.65)
        data_list.extend(coco_evaluator.convert_to_coco_format(
            outputs, info_imgs, ids))

    results = coco_evaluator.evaluate_prediction(data_list)
    print(results)


if __name__ == "__main__":
    main()

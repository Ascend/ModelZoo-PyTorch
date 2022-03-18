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
    output_shape_1 = [1, 4, 80, 80]
    output_shape_2 = [1, 1, 80, 80]
    output_shape_3 = [1, 80, 80, 80]
    output_shape_4 = [1, 4, 40, 40]
    output_shape_5 = [1, 1, 40, 40]
    output_shape_6 = [1, 80, 40, 40]
    output_shape_7 = [1, 4, 20, 20]
    output_shape_8 = [1, 1, 20, 20]
    output_shape_9 = [1, 80, 20, 20]

    input_file_1 = os.path.join(dump_dir, "{:0>12d}_1.bin".format(idx)) 
    input_file_2 = os.path.join(dump_dir, "{:0>12d}_2.bin".format(idx))
    input_file_3 = os.path.join(dump_dir, "{:0>12d}_3.bin".format(idx))
    input_file_4 = os.path.join(dump_dir, "{:0>12d}_4.bin".format(idx))
    input_file_5 = os.path.join(dump_dir, "{:0>12d}_5.bin".format(idx))
    input_file_6 = os.path.join(dump_dir, "{:0>12d}_6.bin".format(idx))
    input_file_7 = os.path.join(dump_dir, "{:0>12d}_7.bin".format(idx))
    input_file_8 = os.path.join(dump_dir, "{:0>12d}_8.bin".format(idx))
    input_file_9 = os.path.join(dump_dir, "{:0>12d}_9.bin".format(idx))
    
    input_data_1 = np.fromfile(input_file_1, dtype=dtype).reshape(output_shape_1)
    input_data_2 = np.fromfile(input_file_2, dtype=dtype).reshape(output_shape_2)
    input_data_3 = np.fromfile(input_file_3, dtype=dtype).reshape(output_shape_3)
    input_data_4 = np.fromfile(input_file_4, dtype=dtype).reshape(output_shape_4)
    input_data_5 = np.fromfile(input_file_5, dtype=dtype).reshape(output_shape_5)
    input_data_6 = np.fromfile(input_file_6, dtype=dtype).reshape(output_shape_6)
    input_data_7 = np.fromfile(input_file_7, dtype=dtype).reshape(output_shape_7)
    input_data_8 = np.fromfile(input_file_8, dtype=dtype).reshape(output_shape_8) 
    input_data_9 = np.fromfile(input_file_9, dtype=dtype).reshape(output_shape_9)
    
    lst = []
    lst.append(torch.from_numpy(input_data_1))
    lst.append(torch.from_numpy(input_data_2))
    lst.append(torch.from_numpy(input_data_3))
    lst.append(torch.from_numpy(input_data_4))
    lst.append(torch.from_numpy(input_data_5))
    lst.append(torch.from_numpy(input_data_6))
    lst.append(torch.from_numpy(input_data_7))
    lst.append(torch.from_numpy(input_data_8))
    lst.append(torch.from_numpy(input_data_9))
    
    return lst


def main():
    parser = argparse.ArgumentParser(description='YOLOX Postprocess')
    parser.add_argument('--dataroot', dest='dataroot',
                        help='data root dirname', default='/opt/npu/coco',
                        type=str)
    parser.add_argument('--dump_dir', dest='dump_dir',
                        help='dump dir for bin files', default='./result/dumpOutput_device0/',
                        type=str)
    
    parser.add_argument('--batch', dest='batch', help='batch for dataloader', default=1, type=int)
    opt = parser.parse_args()
    
    if os.path.exists(opt.dump_dir):
        os.system("rm-rf " + opt.dump_dir)
    else:
        os.system("mkdir " + opt.dump_dir)
    
    valdataset = COCODataset(
        data_dir=opt.dataroot,
        json_file='instances_val2017.json',
        name="val2017",
        img_size = (640, 640),
        preproc=ValTransform(legacy=False),
    )
    sampler = torch.utils.data.SequentialSampler(valdataset)
  
    dataloader_kwargs = {"num_workers": 8, "pin_memory": True, "sampler": sampler, "batch_size": opt.batch}

    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    data_list = []
    coco_evaluator = COCOEvaluator(val_loader, img_size=(640, 640), confthre=0.001, nmsthre=0.65, num_classes=80)
  
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(tqdm(val_loader)):
        
        opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9 = get_output_data(opt.dump_dir, cur_iter)
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

        outputs = postprocess(outputs, num_classes=80, conf_thre=0.001, nms_thre=0.65)
        data_list.extend(coco_evaluator.convert_to_coco_format(outputs, info_imgs, ids)) 

    results = coco_evaluator.evaluate_prediction(data_list)
    print(results)


if __name__ == "__main__":
    main()

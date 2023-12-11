# Copyright 2023 Huawei Technologies Co., Ltd
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
# limitations under the License.ls

import sys
sys.path.append(r'./detr')
import os
import argparse
import time
import numpy as np

import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch_aie
import torchvision.transforms as t
from torch.utils.data import DataLoader

from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models.detr import PostProcess
import transformer as T
from hubconf import detr_resnet50_onnx


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/path/to/coco/')
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--torch_aie_model_dir', type=str, help='/path/to/torch_aie models')
    parser.add_argument("--pre_trained", default="./model/detr.pth", type=str)
    return parser

class TestDetr:

    def __init__(self, args):
        self.args = args
        self.compiled_models_dict_bs1 = {}
        self.compiled_models_dict_bsx = {}
        self.normalize = T.Compose([t.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.detr_transformer = T.Compose([ T.RandomResize([768], max_size=1400), self.normalize,])
        self.device_name = "npu:" + str(self.args.device_id)
        self.stream = torch_aie.npu.Stream(self.device_name)
        self.scalar_to_weight = {"768_1280":0.17, "768_768":0.06, "768_1024":0.53,
                                 "1024_768":0.18, "1280_768":0.05, "768_1344":0.009,
                                 "1344_768":0.0014, "1344_512":0.0006, "512_1344":0.005}

        self.infer_cost_at_scalar = {"768_1280":[], "768_768":[], "768_1024":[],
                                    "1024_768":[], "1280_768":[], "768_1344":[],
                                    "1344_768":[], "1344_512":[], "512_1344":[]}

    def _compile_models(self, batch_size):
        print("start compile models.")
        torch_aie.set_device(self.args.device_id)
        input_shape = [[768, 1280, 24, 40], [768, 768, 24, 24], [768, 1024, 24, 32], [1024, 768, 32, 24],
                       [1280, 768, 40, 24], [768, 1344, 24, 42], [1344, 768, 42, 24], [1344, 512, 42, 16],
                       [512, 1344, 16, 42]]
        for shape in input_shape:
            img_shape = [batch_size, 3, shape[0], shape[1]]
            mask_shape = [batch_size, shape[2], shape[3]]
            scalar_key = str(shape[0]) + "_" + str(shape[1])
            print("start trace model, scalar key is ", scalar_key)
            model = detr_resnet50_onnx(pretrained=False)
            model.load_state_dict(torch.load(args.pre_trained, map_location="cpu")["model"])
            model.eval()
            traced_model = torch.jit.trace(model, (torch.rand(img_shape), torch.zeros(mask_shape, dtype=torch.bool)),
                strict=False)

            torch_aie_model = torch_aie.compile(traced_model,
                                                inputs=[torch_aie.Input(img_shape),
                                                        torch_aie.Input(mask_shape, dtype=torch.bool)],
                                                precision_policy=torch_aie.PrecisionPolicy.FP16)
            self.compiled_models_dict_bsx[scalar_key] = torch_aie_model
            print("compile model success, scalar key is ", scalar_key)

    def get_input_tensor(self, file, need_batch_size = 1):
        img_path = os.path.join(self.args.coco_path, "val2017", file)
        img = cv2.imread(img_path)
        h, w = img.shape[0], img.shape[1]
        input_size = (h, w)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(img)
        img_tensor = self.detr_transformer(pilimg)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        mask_data = torch.zeros([1, int(img_tensor.shape[2] / 32),
                                 int(img_tensor.shape[3] / 32)], dtype=torch.bool)
        if need_batch_size == 1:
                return img_tensor.to(self.device_name), mask_data.to(self.device_name)
        else:
            img_tensor = img_tensor.expand(need_batch_size, -1, -1, -1)
            mask_data =mask_data.expand(need_batch_size, -1, -1,)
            return img_tensor.to(self.device_name), mask_data.to(self.device_name)

    def _load_models(self):
        if len(self.compiled_models_dict_bs1) == 0:
            model_list = os.listdir(self.args.torch_aie_model_dir)
            for model_file in model_list:
                scalar_key = model_file.split(".")[0][5:]
                self.compiled_models_dict_bs1[scalar_key] = \
                    torch.jit.load(os.path.join(args.torch_aie_model_dir, model_file))
                print("load torch_aie model {} success".format(scalar_key))
        else:
            print("torch_aie models have been loaded.")

    def calculate_qps(self, batch_size = 1):
        if batch_size == 1 and self.compiled_models_dict_bs1 is None:
            self._load_models()
            self.compiled_models_dict_bsx = self.compiled_models_dict_bs1
        elif batch_size == 1 and self.compiled_models_dict_bs1 is not None:
            self.compiled_models_dict_bsx = self.compiled_models_dict_bs1
        else:
            self._compile_models(batch_size)
        files = os.listdir('{}/val2017'.format(args.coco_path))
        files.sort()
        for file in tqdm(files):
            img_tensor, mask_data = self.get_input_tensor(file, batch_size)
            scalar_key = str(img_tensor.shape[2]) + "_" + str(img_tensor.shape[3])
            compile_model = self.compiled_models_dict_bsx[scalar_key]
            _, _, scalar_key, time_cost = self.infer(img_tensor, mask_data, compile_model)
            self.infer_cost_at_scalar[scalar_key].append(time_cost)
        avg_step_cost = 0
        for key, val in self.infer_cost_at_scalar.items():
            avg_step_cost += sum(val[1:]) / ( len(val) - 1) * self.scalar_to_weight[key]
        QPS = 1 / avg_step_cost * self.args.batch_size  # 单位s
        print("QPS is {}, bs is {}".format(QPS, batch_size))

    def infer(self, img_tensor, mask_data, compile_model):
        # load compiled models
        with torch_aie.npu.stream(self.stream):
            ts = time.time()
            pred_logits, pred_boxes = compile_model(img_tensor, mask_data)
        self.stream.synchronize()
        te = time.time()
        scalar_key = str(img_tensor.shape[2]) + "_" + str(img_tensor.shape[3])
        pred_logits = pred_logits.to("cpu")
        pred_boxes = pred_boxes.to("cpu")
        return  pred_logits, pred_boxes, scalar_key, te - ts

    def caculate_amp(self):
        torch_aie.set_device(self.args.device_id)
        dataset_val = build_dataset(image_set='val', args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=2)
        base_ds = get_coco_api_from_dataset(dataset_val)
        postprocessors = {'bbox': PostProcess()}
        print('start validate')
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        print_freq = 1000
        files = os.listdir('{}/val2017'.format(args.coco_path))
        files.sort()
        self._load_models()
        print("start calculate amp.")
        for file, (samples, targets) in tqdm(zip(files, metric_logger.log_every(data_loader_val, print_freq, header))):
            img_tensor, mask_data = self.get_input_tensor(file)
            model_name = str(img_tensor.shape[2]) + "_" + str(img_tensor.shape[3])
            compile_model = self.compiled_models_dict_bs1[model_name]
            pred_logits, pred_boxes,_, _ = self.infer(img_tensor, mask_data, compile_model)
            outputs = {'pred_logits': pred_logits,
                      'pred_boxes': pred_boxes}
            metric_logger.update(loss=0.7)
            metric_logger.update(class_error=0.5)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    acc_caculator = TestDetr(args)
    acc_caculator.caculate_amp()
    print("start calculate qps.")
    acc_caculator.calculate_qps(args.batch_size)
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
import time
import os
import json
import sys
from copy import deepcopy

import torch
import torch_aie

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmdeploy.apis.torch_jit import trace

from mmocr.evaluation import HmeanIOUMetric

CURRENT_DIR = os.getcwd()

def get_configs():
    deploy_cfg_file = "./mmdeploy/configs/mmocr/text-detection/text-detection_torchscript.py"
    model_cfg_file = "./mmocr/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
    return load_config(deploy_cfg_file, model_cfg_file)

def get_torch_model(task_processor):
    model_file = "./dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth"
    torch_model = task_processor.build_pytorch_model(model_file)
    torch_model.eval()
    print("built model finish")
    return torch_model

def get_dataloader(task_processor, model_cfg, batch_size):
    test_dataloader = deepcopy(model_cfg['test_dataloader'])
    test_dataloader['batch_size'] = batch_size
    dataset = task_processor.build_dataset(test_dataloader['dataset'])
    test_dataloader['dataset'] = dataset
    dataloader = task_processor.build_dataloader(test_dataloader)
    return dataloader

def preprocess_batch_data(torch_model, batch_data):
    processed_batch_data = torch_model.data_preprocessor(batch_data)
    return processed_batch_data

def get_torchscript_and_aie_model(torch_model, inputs, save_path):
    batch_data = preprocess_batch_data(torch_model, inputs)
    trace_size = list(batch_data['inputs'].size())
    # For tracing DBNet model, we only need to care about HW, they are the same from sample to sample
    trace_size[0] = 1
    with torch.no_grad():
        trace_inputs = torch.rand(trace_size) / 2
        jit_model = trace(torch_model, trace_inputs, output_path_prefix="dbnet", backend='torchscript')
        print("finish trace")

    aie_input_spec=[
        torch_aie.Input(batch_data['inputs'].size()), # Static NCHW input shape for input #1
    ]
    aie_model = torch_aie.compile(jit_model, inputs=aie_input_spec)
    print("aie model compiled")
    aie_model.save(save_path)
    print("aie model saved")

    return jit_model, aie_model

def predict_model(torch_model, batch_data, new_model=None):
    duration = 0
    pred_outputs = None
    if new_model is not None:
        processed_batch_data = preprocess_batch_data(torch_model, batch_data)
        start = time.time()
        outputs = new_model(processed_batch_data['inputs'])
        end = time.time()
        duration = end - start
        pred_outputs = torch_model.det_head.postprocessor(
            outputs, processed_batch_data['data_samples'])
    else:
        with torch.no_grad():
            start = time.time()
            pred_outputs = torch_model.test_step(batch_data)
            end = time.time()
            duration = end - start

    return pred_outputs, duration

def main():
    args = outter_args

    deploy_cfg, model_cfg = get_configs()
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    dataloader = get_dataloader(task_processor, model_cfg, args.batch_size)
    torch_model = get_torch_model(task_processor)

    jit_model = None
    aie_model = None
    aie_model_path = os.path.join(args.aie_model_save_path,  args.aie_model_name)
    if args.trace_compile is True:
        sample_data = next(iter(dataloader))
        jit_model, aie_model = get_torchscript_and_aie_model(
            torch_model, sample_data, aie_model_path)
    else:
        aie_model = torch.jit.load(aie_model_path)
        print("aie model loaded")
    torch_aie.set_device(0)

    aie_metric = HmeanIOUMetric()
    aie_eval_size = 0
    aie_duration = 0

    pytorch_metric = HmeanIOUMetric()
    pytorch_eval_size = 0
    pytorch_duration = 0

    jit_metric = None
    if args.trace_compile is True:
        jit_metric = HmeanIOUMetric()
    jit_eval_size = 0
    jit_duration = 0
    for i, batch_data in enumerate(dataloader):
        print(f'=============================  Batch {i * args.batch_size}  =================================')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx  AIE MODEL  xxxxxxxxxxxxxxxxxxxxxxxxx")
        aie_outputs, duration = predict_model(torch_model, batch_data, new_model=aie_model)
        aie_metric.process(None, aie_outputs)
        aie_eval_size += len(aie_outputs)
        aie_duration += duration
        print("xxxxxxxxxxxxxxxxxxxxxxx  PYTORCH MODEL  xxxxxxxxxxxxxxxxxxxxxxx")
        pytorch_outputs, duration = predict_model(torch_model, batch_data)
        pytorch_metric.process(None, pytorch_outputs)
        pytorch_eval_size += len(pytorch_outputs)
        pytorch_duration += duration
        if args.trace_compile is True:
            print("xxxxxxxxxxxxxxxxxxx  TORCHSCRIPT CPU MODEL  xxxxxxxxxxxxxxxxxxx")
            jit_outputs, duration = predict_model(torch_model, batch_data, new_model=jit_model)
            jit_metric.process(None, jit_outputs)
            jit_eval_size += len(jit_outputs)
            jit_duration += duration
        print(f'===========================================================================')

    print("xxxxxxxxxxxxxxxxxxxxxxxxx  AIE MODEL  xxxxxxxxxxxxxxxxxxxxxxxxx")
    aie_result = aie_metric.evaluate(size=aie_eval_size)
    print(aie_result)
    print(f"qps is: {1 / (aie_duration / aie_eval_size)}")
    print("xxxxxxxxxxxxxxxxxxxxxxx  PYTORCH MODEL  xxxxxxxxxxxxxxxxxxxxxxx")
    pytorch_result = pytorch_metric.evaluate(size=pytorch_eval_size)
    print(pytorch_result)
    print(f"qps is: {1 / (pytorch_duration / pytorch_eval_size)}")
    if args.trace_compile is True:
        print("xxxxxxxxxxxxxxxxxxx  TORCHSCRIPT CPU MODEL  xxxxxxxxxxxxxxxxxxx")
        jit_result = jit_metric.evaluate(size=jit_eval_size)
        print(jit_result)
        print(f"qps is: {1 / (jit_duration / jit_eval_size)}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace_compile',
        action='store_true',
        help='Set True is user want to compile an AIE model, '
        'this will also provide torchscript cpu prodict results.'
    )
    parser.add_argument(
        '--aie_model_name',
        type=str,
        default='aie_model.pt',
        help='Model name for compiled AIE model, end with \".pt\".'
    )
    parser.add_argument(
        '--aie_model_save_path',
        type=str,
        default=CURRENT_DIR,
        help='Model path for saving compiled AIE model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference.'
    )
    outter_args = parser.parse_args()

    if outter_args.trace_compile is False:
        if not os.path.exists(os.path.join(outter_args.aie_model_save_path, outter_args.aie_model_name)):
            print("you choose to not trace and compile an aie_model, \
                  yet you are not giving a valid aie_model file to load")
            sys.exit()

    main()

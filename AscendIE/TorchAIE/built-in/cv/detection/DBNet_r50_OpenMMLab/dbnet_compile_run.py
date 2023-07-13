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

import torch
import torch_aie

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmdeploy.apis.torch_jit import trace

DATA_LIST_NAME = "/data_list.txt"

def get_test_imgs(dataset_root):
    data_list_file = os.path.join(dataset_root, DATA_LIST_NAME)
    img_list = []

    f = open(data_list_file)
    relevant_img_files = f.readlines()
    for relevant_img_file in relevant_img_files:
        relevant_img_file_clean = relevant_img_file.split('\n')[0]
        img_list.append(os.path.join(dataset_root ,relevant_img_file_clean))
    return img_list

def get_torch_model():
    deploy_cfg_file = "./mmdeploy/configs/mmocr/text-detection/text-detection_torchscript.py"
    model_cfg_file = "./mmocr/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
    model_file = "./dbnet_resnet50_1200e_icdar2015_20221102_115917-54f50589.pth"

    deploy_cfg, model_cfg = load_config(deploy_cfg_file, model_cfg_file)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    torch_model = task_processor.build_pytorch_model(model_file)
    torch_model.eval()
    print("build model finish")

    return torch_model, task_processor

def get_inputs(task_processor, torch_model, dataset_root):
    real_imgs = get_test_imgs(dataset_root)
    raw_inputs, _ = task_processor.create_input(real_imgs)  # for pytorch model test_step()
    inputs = torch_model.data_preprocessor(raw_inputs)      # for aie_model and torchscript model
    return inputs, raw_inputs

def get_torchscript_and_aie_model(torch_model, inputs, batch_size, save_path):
    accept_size = list(inputs.size())
    accept_size[0] = 1
    with torch.no_grad():
        trace_inputs = torch.rand(accept_size) / 2
        jit_model = trace(torch_model, trace_inputs, output_path_prefix="dbnet", backend='torchscript')
        print("finish trace")

    accept_size[0] = batch_size
    aie_input_spec=[
        torch_aie.Input(accept_size), # Static NCHW input shape for input #1
    ]
    aie_model = torch_aie.compile(jit_model, inputs=aie_input_spec)
    print("aie model compiled")
    aie_model.save(save_path)
    print("aie model saved")

    return jit_model, aie_model

def get_batch_data(inputs, raw_inputs, idx, batch_size):
    start_idx = idx * batch_size
    end_idx = idx * batch_size + batch_size
    if end_idx > inputs['inputs'].size()[0]:
        end_idx = inputs['inputs'].size()[0]

    pred_inputs = inputs['inputs'][start_idx : end_idx]
    pred_data_samples = inputs['data_samples'][start_idx : end_idx]
    pred_raw_inputs = {
        'inputs' : raw_inputs['inputs'][start_idx : end_idx],
        'data_samples' : raw_inputs['data_samples'][start_idx : end_idx]
    }

    return pred_inputs, pred_data_samples, pred_raw_inputs

def predict_model(torch_model, new_model=None, pred_inputs=None, pred_data_samples=None, pred_raw_inputs=None):
    duration = 0
    pred_outputs = None
    if new_model is not None and pred_inputs is not None and pred_data_samples is not None:
        start = time.time()
        outputs = new_model(pred_inputs)
        end = time.time()
        duration = end - start
        pred_outputs = torch_model.det_head.postprocessor(outputs, pred_data_samples)
    elif not pred_raw_inputs is None:
        with torch.no_grad():
            start = time.time()
            pred_outputs = torch_model.test_step(pred_raw_inputs)
            end = time.time()
            duration = end - start
    print(f"predict outputs are: {pred_outputs}")
    print(f"time cost {duration}")

def main():
    args = outter_args
    torch_model, task_processor = get_torch_model()
    inputs, raw_inputs = get_inputs(task_processor, torch_model, args.dataset_root)

    jit_model = None
    aie_model = None
    aie_model_path = os.path.join(args.aie_model_save_path,  args.aie_model_name)
    if args.trace_compile is True:
        jit_model, aie_model = get_torchscript_and_aie_model(
            torch_model, inputs['inputs'], args.batch_size, aie_model_path)
    else:
        aie_model = torch.jit.load(aie_model_path)
        print("aie model loaded")
    torch_aie.set_device(0)

    for i in range(int(inputs['inputs'].size()[0] / args.batch_size)):
        pred_inputs, pred_data_samples, pred_raw_inputs = get_batch_data(inputs, raw_inputs, i, args.batch_size)
        print(f'=============================  epoch {i}  =================================')
        print("xxxxxxxxxxxxxxxxxxxxxxxxx  AIE MODEL  xxxxxxxxxxxxxxxxxxxxxxxxx")
        predict_model(torch_model, new_model=aie_model, pred_inputs=pred_inputs, pred_data_samples=pred_data_samples)
        print("xxxxxxxxxxxxxxxxxxx  TORCHSCRIPT CPU MODEL  xxxxxxxxxxxxxxxxxxx")
        if args.trace_compile is True:
            predict_model(torch_model, new_model=jit_model,
                pred_inputs=pred_inputs, pred_data_samples=pred_data_samples)
        print("xxxxxxxxxxxxxxxxxxxxxxx  PYTORCH MODEL  xxxxxxxxxxxxxxxxxxxxxxx")
        predict_model(torch_model, pred_raw_inputs=pred_raw_inputs)
        print(f'===========================================================================')

    start = time.time()
    print(aie_model(inputs))
    end = time.time()
    print(f"aie time cost {end - start}")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    start = time.time()
    print(jit_model(inputs))
    end = time.time()
    print(f"jit time cost {end - start}")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    with torch.no_grad():
        start = time.time()
        print(torch_model(inputs))
        end = time.time()
        print(f"pytorch time cost {end - start}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root dir for dataset. The dir must contains a txt file named \"data_list.txt\" '
        'that keeps all relative paths of each image. For example, the root is /data/ and '
        'there is a image /data/dataset/image.jpg then the txt file must contains '
        '\"dataset/image.jpg\" in it on an independent line.'
    )
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
        default='./',
        help='Model path for saving compiled AIE model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference.'
    )
    outter_args = parser.parse_args()

    if not os.path.exists(outter_args.dataset_root):
        print("dataset root is not valid")
        exit
    if not os.path.exists(os.path.join(outter_args.dataset_root, DATA_LIST_NAME)):
        print(f"dataset root does not contains required file \"{DATA_LIST_NAME}\"")
        exit
    if outter_args.trace_compile is False:
        if not os.path.exists(os.path.join(outter_args.aie_model_save_path, outter_args.aie_model_name)):
            print("you choose to not trace and compile an aie_model, \
                  yet you are not giving a valid aie_model file to load")
            exit

    main()

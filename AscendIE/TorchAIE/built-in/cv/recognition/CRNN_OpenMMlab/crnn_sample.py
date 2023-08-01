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

import os
import sys
import time
import copy
import argparse
import json
import torch
import torch_aie

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmocr.apis import TextRecInferencer
from mmdeploy.apis.torch_jit import trace
from mmocr.models.textrecog.postprocessors.ctc_postprocessor import CTCPostProcessor
from mmocr.models.common.dictionary import Dictionary
from mmocr.structures import TextRecogDataSample
from mmengine.structures import LabelData

CURRENT_DIR = os.getcwd()
def get_cfg():
    deploy_cfg_file = "./mmdeploy/configs/mmocr/text-recognition/text-recognition_torchscript.py"
    model_cfg_file = "./mmocr/configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py"
    return load_config(deploy_cfg_file, model_cfg_file)

def get_dataset_info():
    dataset_pre_path = "./mmocr/data/"
    dataset_name = outter_args.dataset
    dataset_path = os.path.join(dataset_pre_path, dataset_name)
    json_file_name = "textrecog_test.json"
    json_file_path = os.path.join(dataset_path, json_file_name)

    test_list = []
    img_path_list = []
    with open(json_file_path, 'r') as json_file:
        dataset_info = json.load(json_file)
        data_list = dataset_info['data_list']
        for image_info in data_list:
            test_list.append(image_info['instances'][0]['text'])
            abs_img_path = os.path.join(dataset_path, image_info['img_path'])
            img_path_list.append(abs_img_path)
    return test_list, img_path_list

# we can get a torch model with task_processor, which is defined by mmdeploy
# the model_ckpt_path can be set  as the real path of pth file
# a mmopenlab-style model contains preprocess and postprocess
def get_torch_model(task_processor):
    if outter_args.model_path == None:
        model_ckpt_path = "./crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth"
    else:
        model_ckpt_path = outter_args.model_path
    torch_model = task_processor.build_pytorch_model(model_ckpt_path)
    if torch_model is not None:
        print("build torch model success")
    return torch_model

# to get a AscendIE model, we shoule build a torchscript model by tracing the torch model
def get_torchs_aie_model(torch_model, deploy_cfg, inputs, data, args):
    data_samples = data['data_samples']
    input_metas = {'data_samples': data_samples, 'mode': 'predict'}
    input_size = list(inputs.size())
    context_info = dict(deploy_cfg=deploy_cfg)
    trace_inputs = torch.rand(input_size)/2
    jit_model = trace(torch_model, trace_inputs, output_path_prefix="crnn", backend='torchscript', 
        input_metas=input_metas, context_info=context_info, check_trace=False)
    input_size[0] = outter_args.batch_size

    # define a static input with the size we want
    compile_input = [torch_aie.Input((input_size))]
    print(f"when compiling the torch aie model, input size is:{input_size}")

    # compile to torch aie model, which can infer by npu
    torch_aie_model = torch_aie.compile(jit_model, inputs=compile_input, allow_tensor_replace_int=True)
    torch_aie.set_device(0)
    if torch_aie_model is not None:
        print("compile model and set device success")
    return torch_aie_model

def get_shaperange_torchs_aie_model(torch_model, deploy_cfg, inputs, data, args):
    data_samples = data['data_samples']
    input_metas = {'data_samples': data_samples, 'mode': 'predict'}
    
    # for shaperange model. create max size and min size to build compile input
    input_max_size = list(inputs.size())
    input_max_size[0] = 1 
    input_min_size = copy.deepcopy(input_max_size)
    input_min_size[-1] = 32
    print(f"when create shaperange input, input_max_size is[{input_max_size}], input_min_size is [{input_min_size}]")
    
    context_info = dict(deploy_cfg=deploy_cfg)
    trace_inputs = torch.rand(input_max_size)/2
    jit_model = trace(torch_model, trace_inputs, output_path_prefix="crnn", backend='torchscript', 
        input_metas=input_metas, context_info=context_info, check_trace=False)

    # define a dynamic input with the minsize and maxsize we want
    shaperange_compile_input = [torch_aie.Input(min_shape = (input_min_size), 
        max_shape = (input_max_size))]
        
    print("start to compile torch_aie model")
    torch_aie_model = torch_aie.compile(jit_model, inputs=shaperange_compile_input,
        allow_tensor_replace_int=True)
    print("compile torch_aie model success")
    
    torch_aie.set_device(0)
    if torch_aie_model is not None:
        print("compile shaperange model and set device success")
    return torch_aie_model, jit_model
    
# a mmocr postprocessor can convert output tensor to text
def get_postprocessor():
    print("start to init postprocessor with mmocr...")
    dict_file = "./mmocr/dicts/lower_english_digits.txt"
    dict_gen = Dictionary(
        dict_file=dict_file,
        with_start=False,
        with_end=False,
        with_padding=True,
        with_unknown=False)
    postprocessor = CTCPostProcessor(max_seq_len=None, dictionary=dict_gen)
    print("init postprocessor with mmocr success")
    return postprocessor

# set min score to flitrate output data
MIN_SCORE = 0.35
def revise_pred_text(pred_text: LabelData, min_score: float) -> LabelData:
    for i in range(len(pred_text.score)):
        while i < len(pred_text.score) and pred_text.score[i] < min_score:
            del pred_text.score[i]
            new_str = ""
            for index in range(0, len(pred_text.item)):
                if index != i:
                    new_str = new_str + pred_text.item[index]
            pred_text.item = new_str
    return pred_text
    
def test_dataset_shaperange():
    args = outter_args
    deploy_cfg, model_cfg = get_cfg()
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    test_list, img_path_list = get_dataset_info()
    torch_model = get_torch_model(task_processor)
    torch_model.eval()
    
    data, _ = task_processor.create_input(img_path_list)
    data_samples = data['data_samples']
    inputs = torch_model.data_preprocessor(data)['inputs']
    shaperange_torch_aie_model, jit_model = \
        get_shaperange_torchs_aie_model(torch_model, deploy_cfg, inputs, data, args)
        
    postprocessor = get_postprocessor()
    total_aie_time = 0
    total_torch_time = 0
    
    if img_path_list is not None:
        total_test_num = 0
        aie_matched_num = 0
        torch_matched_num = 0
        
        for img in img_path_list:
            single_img_data, _ = task_processor.create_input([img])
            data_sample = single_img_data['data_samples']
            inputs = torch_model.data_preprocessor(single_img_data)['inputs']
            start = time.time()
            aie_result = shaperange_torch_aie_model(inputs)
            end = time.time()
            total_aie_time += end - start
            
            aie_data_samples = postprocessor(aie_result, data_sample)
            pred_text = aie_data_samples[0].pred_text
            total_test_num = total_test_num + 1
            
            if pred_text.item.lower() == test_list[total_test_num - 1].lower():
                aie_matched_num = aie_matched_num + 1
            
            jit_result = jit_model(inputs)
            jit_data_samples = postprocessor(jit_result, data_sample)
            jit_pred_text = jit_data_samples[0].pred_text
            
            start = time.time()
            torch_result = torch_model(inputs, data_sample)
            end = time.time()
            total_torch_time += end - start
            pred_text = torch_result[0].pred_text
            
            if pred_text.item.lower() == test_list[total_test_num - 1].lower():
                torch_matched_num = torch_matched_num + 1
        
        print(f"time cost of aie model is[{total_aie_time}]")
        print(f"time cost of torch model is[{total_torch_time}]")
        print(f"total test num is: {total_test_num}, aie match num is:{aie_matched_num}, torch match num is:{torch_matched_num}")
            
             

def test_dataset():
    args = outter_args
    deploy_cfg, model_cfg = get_cfg()
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    test_list, img_path_list = get_dataset_info()
    torch_model = get_torch_model(task_processor)
    torch_model.eval()

    data, _ = task_processor.create_input(img_path_list)
    data_samples = data['data_samples']
    inputs = torch_model.data_preprocessor(data)['inputs']
    torch_aie_model = get_torchs_aie_model(torch_model, deploy_cfg, inputs, data, args)

    postprocessor = get_postprocessor()
    batchsize = args.batch_size
    total_aie_time = 0
    total_torch_time = 0

    if img_path_list is not None:
        total_test_num = 0
        aie_matched_num = 0
        torch_matched_num = 0
        for i in range(inputs.size()[0] // batchsize):
            start_index = i * batchsize
            if start_index + batchsize >= inputs.size()[0]:
                input_data = inputs[start_index : ]
                current_data_samples = data_samples[start_index : ]
                current_result = test_list[start_index : ]
            else:
                input_data = inputs[start_index : start_index + batchsize]
                current_data_samples = data_samples[start_index : start_index + batchsize]
                current_result = test_list[start_index : start_index + batchsize]
            start = time.time()
            aie_result = torch_aie_model(input_data)
            end = time.time()
            total_aie_time += end - start
            aie_data_samples = postprocessor(aie_result, current_data_samples)
            for pic in range(batchsize):
                pred_text = aie_data_samples[pic].pred_text
                revise_pred_text(pred_text, MIN_SCORE)
                total_test_num = total_test_num + 1
                if pred_text.item.lower() == current_result[pic].lower():
                    aie_matched_num = aie_matched_num + 1

            start = time.time()
            torch_result = torch_model(input_data, current_data_samples)
            end = time.time()
            total_torch_time += end - start
            for pic in range(batchsize):
                pred_text = torch_result[pic].pred_text
                revise_pred_text(pred_text, MIN_SCORE)
                if pred_text.item.lower() == current_result[pic].lower():
                    torch_matched_num = torch_matched_num + 1
        print(f"total test num is: {total_test_num}, aie match num is:{aie_matched_num}, \
            torch match num is:{torch_matched_num}")
    print(f"aie time cost is: {total_aie_time}")
    print(f"torch time cost is: {total_torch_time}")


def test_img():
    args = outter_args
    test_imgs = args.img_path
    deploy_cfg, model_cfg = get_cfg()
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    torch_model = get_torch_model(task_processor)
    torch_model.eval()

    img_list = [test_imgs]
    data, _ = task_processor.create_input(img_list)
    inputs = torch_model.data_preprocessor(data)['inputs']

    data_samples = data['data_samples']
    input_metas = {'data_samples': data_samples, 'mode': 'predict'}
    context_info = dict(deploy_cfg=deploy_cfg)

    input_size = list(inputs.size())
    trace_inputs = torch.rand(input_size)/2
    jit_model = trace(torch_model, trace_inputs, output_path_prefix="crnn", backend='torchscript', 
        input_metas=input_metas, context_info=context_info, check_trace=False)
    jit_model.save("crnn_ts.pth")

    compile_input = [torch_aie.Input((input_size))]
    torch_aie_model = torch_aie.compile(jit_model, inputs=compile_input, allow_tensor_replace_int=True)
    torch_aie.set_device(0)
    if torch_aie_model is not None:
        print("compile model and set device success")
    
    postprocessor = get_postprocessor()
    start = time.time()
    aie_result = torch_aie_model(inputs)
    end = time.time()
    aie_data_sample = postprocessor(aie_result, data_samples)
    perd_text = aie_data_sample[0].pred_text
    print(f"the recognition result of aie model is [{perd_text.item}], time use is [{end - start}]")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace_compile',
        type=bool,
        default=True,
        help='Set True if user want to compile an AIE model, '
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path of the pth model'
    )
    parser.add_argument(
        '--aie_model_name',
        type=str,
        default='aie_model.pt',
        help='Model name for compiled AIE model, end with \".pt\". Not supported yet'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='dataset name for test.'
    )
    parser.add_argument(
        '--aie_model_save_path',
        type=str,
        default=CURRENT_DIR,
        help='Model path for saving compiled AIE model. Not supported yet'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        default=None,
        help='Image path for test.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference.'
        'default bs is 1, the same as mmocr did.'
    )
    parser.add_argument(
        '--shape_range',
        type=bool,
        default=False,
        help='Whether to use dynamic input size. '
        'Only support the dataset for multi-image recognition. Not supported yet'
    )
    outter_args = parser.parse_args()

    if outter_args.trace_compile is False:
        if not os.path.exists(os.path.join(outter_args.aie_model_save_path, outter_args.aie_model_name)):
            print("you choose to not trace and compile an aie_model, \
                  yet you are not giving a valid aie_model file to load")
            sys.exit()
    
    if outter_args.img_path is None and outter_args.dataset is None:
        print("Attention! the --dataset and --img_path are both empty, please enter one of them for test!")
        sys.exit()
    
    if outter_args.shape_range and outter_args.dataset is not None:
        test_dataset_shaperange()
    elif outter_args.dataset is not None:
        test_dataset()
    elif outter_args.img_path is not None:
        test_img()    

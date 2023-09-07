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
# limitations under the License.

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import torch_npu
import platform
import subprocess
import os
from colorama import Fore, Style
from tempfile import NamedTemporaryFile

SEQ_LEN_OUT = 64

# set npu device
DEVICE_ID = 0
torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

# set model path
MODEL_PATH = "./"
print(">>>> running BAICHUAN2_13B <<<<<<<")


def remove_part_of_generation_config(generation_config):
    """
    移除部分后处理相关参数，预期9月底支持
    :param generation_config:
    :return:
    """
    ori_gen = GenerationConfig()
    diff_dict = generation_config.to_diff_dict()
    print(diff_dict)
    for key in diff_dict:
        if key.endswith("_id"):
            continue
        ori_value = getattr(ori_gen, key, None)
        if ori_value is not None:
            setattr(generation_config, key, getattr(ori_gen, key))
            print(f"replace {key}")
    return generation_config


def trans_data_format(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(2)
        print("soc version: ", soc_version, " is 910B, support ND")
    else:
        # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = module.weight.data.npu_format_cast(29)
        print("soc version: ", soc_version, " is not 910B, support NZ")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.npu_format_cast(2)


def init_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    print(">>>> tokenizer", tokenizer)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(">>>> load model begin", config)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True)
    model = model.half().npu()
    model.generation_config = remove_part_of_generation_config(model.generation_config)
    # trans data format
    trans_data_format(model)

    return model, tokenizer


def run_infer_test(model, tokenizer):
    # warm-up using huggingface's generate api
    print("--------------warm up--------------")
    test_prompt = "你是谁？请介绍。"
    inputs_warm_up = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        pred1 = model.generate(inputs_warm_up.input_ids.npu(), attention_mask=inputs_warm_up.attention_mask.npu(),
                               max_new_tokens=SEQ_LEN_OUT)
        print(tokenizer.decode(pred1[0], skip_special_tokens=True))

    # inference using huggingface's generate api
    print("--------------inference--------------")
    inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors="pt")

    start_time = time.time()
    with torch.no_grad():
        pred2 = model.generate(inputs.input_ids.npu(), attention_mask=inputs.attention_mask.npu(),
                               max_new_tokens=SEQ_LEN_OUT)
    end_time = time.time()

    # decode
    print(tokenizer.decode(pred2[0], skip_special_tokens=True))

    # time analysis
    new_tokens = len(pred2[0]) - len(inputs.input_ids[0])
    elapse = end_time - start_time
    print(f"[Output tokens number]: {len(pred2[0])}, \n[Input tokens number]: {len(inputs.input_ids[0])},\n[total new tokens generated]: {new_tokens}")
    print(f"Output generated in {elapse:.2f}s, {(new_tokens / elapse):.2f} tokens/s, {new_tokens} new tokens generated.")


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用百川大模型，输入进行对话，vim 多行输入，clear 清空历史，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def run_chat(model, tokenizer, stream=True):
    print(">>>> start chat")
    messages = []
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBaichuan：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True, device_id=DEVICE_ID):
                    print(response[position:], end='', flush=True)
                    position = len(response)
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == '__main__':
    model, tokenizer = init_model(MODEL_PATH)
    # run infer test
    run_infer_test(model, tokenizer)
    # run chat test, input 'exit' to exit
    run_chat(model, tokenizer)

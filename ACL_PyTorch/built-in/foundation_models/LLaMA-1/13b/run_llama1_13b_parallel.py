# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import time
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
import torch_npu
from torch_npu.contrib import transfer_to_npu

SEQ_LEN_IN = 128
SEQ_LEN_OUT = 128


def setup_model_parallel():
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank == 0:
        torch_npu.npu.set_device(0)
    elif local_rank == 1:
        torch_npu.npu.set_device(1)
    torch.manual_seed(1)
    return local_rank, world_size


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


def init_model(init_args):
    tokenizer_path = os.path.join(init_args.load_path, '/tokenizer')
    tokenizer_init = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    tokenizer_init.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    print("load tokenizer success!")

    part_model_path = os.path.join(init_args.load_path, '/part_model/', str(local_rank), '/')
    model_init = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
    model_init.resize_token_embeddings(len(tokenizer_init))  # pad or not
    print("load model success!")
    
    trans_data_format(model_init)

    return model_init, tokenizer_init


def warm_up(warmup_model, warmup_tokenizer):
    print("warm-up start")
    test_prompt = "Common sense questions and answers\n\nQuestion: Why do people need sleep\nFactual answer:"
    inputs_warm_up = warmup_tokenizer(test_prompt, return_tensors="pt")
    _ = warmup_model.generate(
        inputs_warm_up.input_ids.npu(),
        max_new_tokens=SEQ_LEN_OUT
    )
    torch.npu.empty_cache()
    print("warm-up success!")


def inference(infer_model, infer_tokenizer):
    print("inference start")
    prompt = [
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
    ]
    # tokenize
    start_time = time.time()
    inputs = infer_tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=SEQ_LEN_IN)
    #infer
    with torch.no_grad():
        generate_ids = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT)
    end_time = time.time()
    print("inference success!")
    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # output
    for item in res:
        print(item)
    # time analysis
    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    print(f"Input tokens number: {len(inputs.input_ids[0])},\ntotal new tokens generated: {new_tokens}")
    print(f"Output generated in {(end_time-start_time):.2f} s ({new_tokens/(end_time-start_time):.2f} tokens/s")


if __name__ == "__main__":
    # initialize tensor-parallel mode
    local_rank, world_size = setup_model_parallel()

    # adapt torch-npu
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    # load path
    parser = argparse.ArgumentParser(
        description="load Model weights and run")
    parser.add_argument(
        "--load_path",
        default="/data/models/llama-13b-part_model_2",
        help="Location of Model weights, which contains model folders",
    )
    args = parser.parse_args()

    # load tokenizer and model
    model, tokenizer = init_model(args)

    # warm up
    warm_up(model, tokenizer)

    # inference
    inference(model, tokenizer)







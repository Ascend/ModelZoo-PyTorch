# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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
import torch_npu
from fastchat.serve.inference import load_model,ChatIO
from fastchat.utils import get_context_length
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.conversation import (
    conv_templates,
    get_conv_template
)

class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                #print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        #print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)


def load(model_path,device,num_gpus):
    model,tokenizer = load_model(model_path,device,num_gpus)
    return model,tokenizer
def Chat(model,tokenizer,model_path,device,inp):
    conv = get_conv_template("zero_shot")
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #skip_echo_len = compute_skip_echo_len(model_path, conv , prompt)
    gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": 0.9,
            "max_new_tokens": 2048,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
    chatio = SimpleChatIO()
    context_len = get_context_length(model.config)
    generate_stream_func = get_generate_stream_function(model, model_path)
    output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
            )
    #output_stream = generate_stream_func(model, tokenizer, params, device)
    outputs = chatio.stream_output(output_stream)
    return outputs,0
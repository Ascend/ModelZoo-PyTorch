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

import os, argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

import torch_aie
from torch_aie import _enums


def build_tokenizer(tokenizer_name):
    tokenizer_kwargs = {
        'cache_dir': None,
        'use_fast': True,
        'revision': 'main',
        'use_auth_token': None
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    return tokenizer

def build_base_model(tokenizer, model_path, config_path, device):
    config_kwargs = {
        'cache_dir': None,
        'revision': 'main',
        'use_auth_token': None
    }
    config = AutoConfig.from_pretrained(config_path, **config_kwargs)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        config=config,
        revision='main',
        use_auth_token=None
    )
    model.to(device=device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    return model

class RefineModel(torch.nn.Module):
    def __init__(self, tokenizer, model_path, config_path, device="cpu"):
        super(RefineModel, self).__init__()
        self._base_model = build_base_model(tokenizer, model_path, config_path, device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self._base_model(input_ids, attention_mask, token_type_ids)
        return x[0].argmax(dim=-1)
    
def compute_metrics(preds, labels, eval_metric_path="./accuracy.py"):
    metric = load_metric(eval_metric_path)
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask] # 返回labels当中值不等于 -100的所有元素组成的列表
    preds = preds[mask]

    return metric.compute(predictions=preds, references=labels)


def load_data(input_path, shape, dtype="int64"):
    return np.fromfile(input_path, dtype=dtype).reshape(shape)


def load_datasets(input_dir, gt_dir, batch_size, seq_length):
    input_ids_vec_list = []
    attention_mask_list = []
    token_type_ids_list = []
    all_labels_list = []

    num_datas = len(os.listdir(gt_dir)) # 真实值：12800

    for step in tqdm(range(num_datas)):
        input_bin_path = os.path.join(input_dir, "input_ids", "{}.bin".format(step))
        input_ids_vec = np.fromfile(input_bin_path, dtype=np.int64).reshape((batch_size, seq_length))
        input_ids_vec = torch.tensor(input_ids_vec, dtype=torch.int64)
        input_ids_vec_list.append(input_ids_vec)
        
        mask_bin_path = os.path.join(input_dir, "attention_mask", "{}.bin".format(step))
        attention_mask = np.fromfile(mask_bin_path, dtype=np.int64).reshape((batch_size, seq_length))
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        attention_mask_list.append(attention_mask)
        
        token_type_path = os.path.join(input_dir, "token_type_ids", "{}.bin".format(step))
        token_type_ids = np.fromfile(token_type_path, dtype=np.int64).reshape((batch_size, seq_length))
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)
        token_type_ids_list.append(token_type_ids)

        label_path = os.path.join(gt_dir, "{}.bin".format(step))
        labels = load_data(label_path, [batch_size, seq_length])
        all_labels_list.append(labels)

    datasets = {'input_ids_vec_list': input_ids_vec_list,
                    'attention_mask_list': attention_mask_list,
                    'token_type_ids_list': token_type_ids_list,
                    'all_labels_list': all_labels_list,
                    'num_datas': num_datas}
    return datasets


def inference(torchaie_model, datasets):
    input_ids_vec_list = datasets['input_ids_vec_list']
    attention_mask_list = datasets['attention_mask_list']
    token_type_ids_list = datasets['token_type_ids_list']
    num_datas = datasets['num_datas']
    
    all_logits_list = []
    inference_time = []
    count = 0
    for step in tqdm(range(num_datas)):
        input_ids_vec_npu = input_ids_vec_list[step].to('npu:0')
        attention_mask_npu = attention_mask_list[step].to('npu:0')
        token_type_ids_npu = token_type_ids_list[step].to('npu:0')

        stream = torch_aie.npu.Stream('npu:0')
        with torch_aie.npu.stream(stream):
            inf_start = time.time()
            logits = torchaie_model(input_ids_vec_npu, attention_mask_npu, token_type_ids_npu)
            stream.synchronize()
            inf_end = time.time()
            inf_time = inf_end - inf_start
            count = count + 1
            if count > 5:
                inference_time.append(inf_time)
            logits = logits.to('cpu')
            all_logits_list.append(logits)

    average_inference_time = np.mean(inference_time)
    print('pure model inference performance=', average_inference_time * 1000, 'ms')
    return all_logits_list
    

def run(torchaie_model, input_dir, gt_dir, batch_size, seq_length):
    datasets = load_datasets(input_dir, gt_dir, batch_size, seq_length)
    num_datas = datasets['num_datas']
    all_labels_list = datasets['all_labels_list']

    start = time.time()
    all_logits_list = inference(torchaie_model, datasets)
    end = time.time()
    times = end - start
    performance = times / num_datas
    print('end2end performance=', performance * 1000, 'ms')

    all_logits_array = np.concatenate(all_logits_list, axis=0)
    all_labels_array = np.concatenate(all_labels_list, axis=0)

    metric = compute_metrics(all_logits_array, all_labels_array)
    print('metric=', metric)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--input_dir", type=str, default="./input_data")
    parser.add_argument("--gt_dir", type=str, help="Path to the model directory", default="./input_data/labels")
    parser.add_argument("--seq_length", type=int, help="Sequence length as an integer", default=384)
    parser.add_argument('--ts_model_path', type=str, default='bert_base_chinese.pt',
                            help='Original TorchScript model path')

    args = parser.parse_args()
    
    input_dir = args.input_dir
    gt_dir = args.gt_dir
    seq_length = args.seq_length
    ts_model = torch.jit.load(args.ts_model_path)
    
    torch_aie.set_device(0)
    input_info = [torch_aie.Input((1, 384), dtype=torch.int64),
                    torch_aie.Input((1, 384), dtype=torch.int64),
                    torch_aie.Input((1, 384), dtype=torch.int64)]

    torchaie_model = torch_aie.compile(
        ts_model,
        inputs=input_info,
        precision_policy=torch_aie.PrecisionPolicy.PREF_FP32, # 这里必须使用混合精度才不会掉精度！
        truncate_long_and_double=True,
        require_full_compilation=True,
        allow_tensor_replace_int=True,
        torch_executed_ops=[],
        soc_version="Ascend310P3",
        optimization_level=0,
    )

    torchaie_model.eval()
    
    batch_size = 1
    run(torchaie_model, input_dir, gt_dir, batch_size, seq_length)

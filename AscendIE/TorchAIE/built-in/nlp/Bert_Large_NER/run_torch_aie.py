import os
import time
import copy
import argparse
from pathlib import Path

import torch
import torch_aie
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

NPU_DEVICE = 'npu:0'
NAME_ENTITY= {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}


def get_test_datasets(test_file_path, model_config_path):
    config_path = os.path.abspath(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(config_path)
    test_file = Path(test_file_path)
    if not test_file.is_file():
        print('Test dataset files not exist')
        exit(0)
    lines = []
    with test_file.open() as infile:
        for line in tqdm(infile):
            lines.append(line)
    sentence_words, sentence_tags = [], []
    input_ids, token_type_ids, attention_mask = [], [], []
    item_num = 0
    tags_total = []
    input_ids_total, token_type_ids_total, attention_mask_total = [], [], []
    for line in lines:
        line = line.strip('\n')
        if line == '-DOCSTART- -X- -X- O':
            continue
        if line == '':
            if len(sentence_words):
                for i in range(len(sentence_words)):
                    if sentence_words[i].isupper():
                        sentence_words[i]=sentence_words[i].lower()
                tokens = tokenizer.convert_tokens_to_ids(sentence_words)
                input_ids.append(np.pad(tokens, (0, 512-len(tokens)), 'constant'))
                token_type_ids.append([0]*512)
                attention_mask.append([1]*len(tokens)+[0]*(512-len(tokens)))
                tags = []
                for tag in sentence_tags:
                    tags.append(NAME_ENTITY[tag])
                tags = np.array(tags)
                item_num += len(tags)
                input_ids_total.append(copy.deepcopy(input_ids))
                token_type_ids_total.append(copy.deepcopy(token_type_ids))
                attention_mask_total.append(copy.deepcopy(attention_mask))
                tags_total.append(copy.deepcopy(tags))
                sentence_words.clear()
                sentence_tags.clear()
                input_ids.clear()
                token_type_ids.clear()
                attention_mask.clear()
        else:
            word, _, _, tag = line.split(' ')
            sentence_words.append(word)
            sentence_tags.append(tag)
    test_dataset = {'input_ids': input_ids_total,
                    'token_type_ids': token_type_ids_total,
                    'attention_masks': attention_mask_total,
                    'tags': tags_total,
                    'item_num': item_num
                    }
    return test_dataset


def get_torchaie_model(ts_model_path):
    ts_model = torch.jit.load(ts_model_path)
    ts_model.eval()

    input_info = [torch_aie.Input((1, 512), dtype=torch.int64),
        torch_aie.Input((1, 512), dtype=torch.int64),
        torch_aie.Input((1, 512), dtype=torch.int64)]
    
    torch_aie.set_device(0)
    torchaie_model = torch_aie.compile(
        ts_model,
        inputs=input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False, # graph的最后一个算子输出是tuple类型的，不支持导出整图
        allow_tensor_replace_int=True,
        torch_executed_ops=[],
        soc_version="Ascend310P3",
        optimization_level=0,
    )
    torchaie_model.eval()
    return torchaie_model


def run_test(torchaie_model, test_dataset):
    correct, count = 0, 0
    item_num = test_dataset['item_num']
    inference_times = []
    for i in range(len(test_dataset['input_ids'])):
        input_id = test_dataset['input_ids'][i]
        token_type_id = test_dataset['token_type_ids'][i]
        attention_mask = test_dataset['attention_masks'][i]
        tag = test_dataset['tags'][i]

        input_ids_npu = torch.tensor(np.array(input_id)).to(NPU_DEVICE)
        token_type_ids_npu = torch.tensor(np.array(token_type_id)).to(NPU_DEVICE)
        attention_mask_npu = torch.tensor(np.array(attention_mask)).to(NPU_DEVICE)
        stream = torch_aie.npu.Stream(NPU_DEVICE)
        with torch_aie.npu.stream(stream):
            inf_start = time.time()
            result = torchaie_model(input_ids_npu, token_type_ids_npu, attention_mask_npu)
            stream.synchronize()
            inf_end = time.time()
            inf_time = inf_end - inf_start
            count = count + 1
            if count > 5:
                inference_times.append(inf_time)
            result_cpu = result[0].to('cpu')
            pred = result_cpu.argmax(axis=2)
            pred = pred[0].tolist()[:len(tag)]
            correct += (tag == pred).sum()
    acc = correct / item_num
    print(f'Accuracy={acc * 100: .2f}%')
    average_inference_time = np.mean(inference_times)
    print('Pure model inference performance=', average_inference_time * 1000, 'ms')
    print('Performance=', 1.0 / average_inference_time, 'it/s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert-large-NER run on torch-aie.')
    parser.add_argument('--ts_model_path', type=str, default='./bert_large_ner.pt',
                    help='bert-large-NER run on torch-aie')
    parser.add_argument('--model_config_path', type=str, default='./bert-large-NER',
                help='huggingface local model config path')
    parser.add_argument('--test_file_path', type=str, default='./conll2003/test.txt',
                help='test dataset txt file path')
    args = parser.parse_args()

    torchaie_model = get_torchaie_model(args.ts_model_path)
    test_dataset = get_test_datasets(args.test_file_path, args.model_config_path)
    run_test(torchaie_model, test_dataset)
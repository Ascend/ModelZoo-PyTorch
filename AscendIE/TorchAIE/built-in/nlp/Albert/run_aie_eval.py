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

from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, '../')
import time
import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
# from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification # chinese version
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar

import torch_aie


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) 
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn, drop_last=True)

        # Eval!
        infer_stream = torch_aie.npu.Stream("npu:0")
        h2d = torch_aie.npu.Stream("npu:0")

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        preds = None
        out_label_ids = None
        fps = []

        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            # prepare data on npu
            with torch_aie.npu.stream(h2d):
                b = []
                for t in batch:
                    t_npu = t.to(args.device)
                    b.append(t_npu)
                inputs = {'input_ids': b[0],
                          'attention_mask': b[1],
                          'labels': b[3]}
                inputs['token_type_ids'] = b[2]
            h2d.synchronize()
            # forward
            with torch.no_grad():
                inf_s = time.time()
                with torch_aie.npu.stream(infer_stream):
                    outputs = model(b[0], b[1], b[2])
                infer_stream.synchronize()
                inf_e = time.time()
                fps.append(args.eval_batch_size / (inf_e - inf_s))
                logits = outputs[:1]
            # process preds
            if preds is None:
                preds = logits[0].detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits[0].detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        throughput = sum(fps[1:]) / len(fps[1:])
        avg_inf_time = args.eval_batch_size / throughput
        print("Throughput is {} and average inference time cost is {}".format(throughput, avg_inf_time))

    return results

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode = output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int32)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int32)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.int32)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.int32)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.int32)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--aie_model_dir", default=None, type=str, required=True,
                        help="Path to compiled torch aie model")                    
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--vocab_file",default='', type=str)
    parser.add_argument("--spm_model_file",default='',type=str)

    ## Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))

    # setup npu device
    torch_aie.set_device(0)
    device = torch.device("npu:0")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, torch.device("cpu"), args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    seed_everything(args.seed)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    logger.info("Evaluation parameters %s", args)

    # Evaluation
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file,
                                                      do_lower_case=args.do_lower_case,
                                                      spm_model_file=args.spm_model_file)
        checkpoints = [(0,args.output_dir)]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]),checkpoint) for checkpoint in checkpoints if checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints,key =lambda x:x[0])
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _,checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = torch.jit.load(args.aie_model_dir)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key,value in results:
                writer.write("%s = %s\n" % (key, str(value)))


if __name__ == "__main__":
    main()
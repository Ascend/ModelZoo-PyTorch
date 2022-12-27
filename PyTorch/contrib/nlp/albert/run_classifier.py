# encoding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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

""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import glob

import numpy as np
import apex
import torch
if torch.__version__ >= "1.8":
    import torch_npu
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from callback.optimization.lamb import Lamb
from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar
from tools.fps_counter import FpsCounter

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29688'
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
NUM_WORKER = 16
WORLD_SIZE = 8


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn,# 动态shape罪魁祸首
                                  drop_last=True,
                                  num_workers=args.num_workers)  # add worker

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if 'npu' in str(args.device):
        optimizer = apex.optimizers.NpuFusedLamb(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if 'npu' in str(args.device):
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, loss_scale="dynamic",
                                              combine_grad=True)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level, loss_scale=128.0)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in range(int(args.num_train_epochs)):
        if args.local_rank > -1:
            train_sampler.set_epoch(epoch)  # add via hw
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        # fps
        fps = FpsCounter()
        for step, batch in enumerate(train_dataloader):
            print('=====iter%d' % global_step)
            if args.local_rank == -1 and step == 30:  # 单卡模式生成prof
                if 'npu' in str(args.device):
                    with torch.autograd.profiler.profile(use_npu=False if 'cpu' in str(args.device) else True) as prof:
                        model.train()
                        batch = tuple(t.to(args.device) for t in batch)
                        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                        inputs['token_type_ids'] = batch[2]

                        outputs = model(**inputs)
                        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu parallel training
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        tr_loss += loss.item()
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            scheduler.step()  # Update learning rate schedule
                            model.zero_grad()
                            global_step += 1
                    logger.info("npu profiler dump to {}{}npu.prof".format(args.output_dir, args.model_type))
                    prof.export_chrome_trace("{}{}npu.prof".format(args.output_dir, args.model_type))
                else:
                    with torch.autograd.profiler.profile(use_cuda=False if 'cpu' in str(args.device) else True) as prof:
                        model.train()
                        batch = tuple(t.to(args.device) for t in batch)
                        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                        inputs['token_type_ids'] = batch[2]

                        outputs = model(**inputs)
                        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu parallel training
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        tr_loss += loss.item()
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            scheduler.step()  # Update learning rate schedule
                            model.zero_grad()
                            global_step += 1
                    logger.info("cuda profiler dump to {}{}gpu.prof".format(args.output_dir, args.model_type))
                    prof.export_chrome_trace("{}/{}.prof".format(args.output_dir, args.model_type))
            else:
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]

                if step > 5:
                    fps.begin()

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                if step > 5:
                    fps.end()

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                evaluate(args, model, tokenizer)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
            pbar(step, {'loss': loss.item()})

        print(" ")

        fps_val = fps.fps(args.batch_size, args.n_gpu)
        if args.local_rank == -1:
            logger.info(" FPS = {:.2f}".format(fps_val))
        else:
            torch.save(fps_val, './fps%d' % args.local_rank)
        fps.reset()

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if 'npu' in str(args.device):
            torch.npu.empty_cache()

        if args.local_rank == 0:
            fpss = []
            for i in range(WORLD_SIZE):
                fpss.append(torch.load('./fps%d' % i))
            m = sum(fpss)
            logger.info(" FPS = {:.2f}".format(m))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    global OK
    # 防止出错，只用单卡有序跑
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn,# 动态shape罪魁祸首
                                     drop_last=True,
                                     num_workers=0)  # add worker x 容易出错

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

        for step, batch in enumerate(eval_dataloader):
            model.eval()
            if batch[0].size()[0] != args.batch_size: continue
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            if preds is None:
                preds = logits.float().detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.float().cpu().detach().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if 'npu' in str(args.device):
            torch.npu.empty_cache()

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset
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
                                                output_mode=output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--loss_scale", default=128.0, type=float, required=False, help="")
    parser.add_argument("--device", default="cuda", type=str, required=False, help="cuda/cpu/npu")
    parser.add_argument("--data_dir", default="./dataset/glue/SST-2/", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="SST", type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default="./outputs/glue/SST", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--vocab_file", default='', type=str)
    parser.add_argument("--spm_model_file", default='', type=str)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10%")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--device-id", type=int, default=0, help="1p special card")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--stop_point", default=1.0, type=float,
                        help="early stop")
    args = parser.parse_args()

    os.environ['RANK'] = "%d" % args.local_rank

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.environ['WORLD_SIZE'] = "%d" % WORLD_SIZE
    # Setup CUDA, GPU & distributed training #rank -1不用，0主机，正整数从机
    if args.local_rank == -1 or args.no_cuda:
        if args.device == 'npu':
            device = torch.device(f"npu:{args.device_id}" if torch.npu.is_available() and not args.no_cuda else "cpu")
        else:
            device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        args.n_gpu = 1
        torch.npu.set_device(args.device_id)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if args.device == 'npu':
            torch.npu.set_device(args.local_rank)
            device = torch.device("npu", args.local_rank)
            torch.distributed.init_process_group(backend='hccl')
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.num_workers = NUM_WORKER
    ##################### set global data into logging
    logger.args = args

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AlbertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                  spm_model_file=args.spm_model_file)
    model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                                            config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # reset early stop
    if os.path.exists('./ok'):
        os.remove('./ok')

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file,
                                                      do_lower_case=args.do_lower_case,
                                                      spm_model_file=args.spm_model_file)
        checkpoints = [(0, args.output_dir)]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]), checkpoint) for checkpoint in checkpoints if
                           checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints, key=lambda x: x[0])
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _, checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = AlbertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key, value in results:
                writer.write("%s = %s\n" % (key, str(value)))


if __name__ == "__main__":
    main()

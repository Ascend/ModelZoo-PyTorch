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

"""TinyBERT finetuning runner specifically for SST-2 dataset."""

################## import libraries ##################

#standard libraries
from __future__ import absolute_import, division, print_function
import sys
import argparse
import csv
import logging
import os
import random
import time

#third-party libraries
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.npu
from apex import amp
from torch.nn import CrossEntropyLoss, MSELoss
import torch.multiprocessing as mp

#local libraries
from transformer.modeling import TinyBertForSequenceClassification
from transformer.modeling_for_finetune import TinyBertForSequenceClassification_for_finetune
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

################## end import libraries ##################

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./output/debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

################## define classes and functions ##################

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels.cpu())
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(features):
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, args, step, rank, delta_time, epoch, ngpus_per_node):
    if (step + 1) % args.output_txt_step == 0:
        if args.pred_distill:
            basic = "[NPU:" + str(rank) + "] "
            epoch = ' epoch ' + str(epoch)
            step = ' step' + str(step)
            fps = 'FPS{} '.format(args.eval_batch_size * ngpus_per_node / delta_time)
            acc = 'acc{} '.format(result['acc'])
            content = basic + epoch + step + fps + acc + "\n"
            if args.performance:
                file_name = args.fps_acc_dir + '/train_performance_' + str(ngpus_per_node) + 'p_2.txt'
            else:
                file_name = args.fps_acc_dir + '/train_full_' + str(ngpus_per_node) + 'p_2.txt'
            with open(file_name, "a") as writer:
                writer.write(content)
        else:
            basic = "[NPU:" + str(rank) + "] "
            epoch = ' epoch ' + str(epoch)
            step = ' step' + str(step)
            fps = ' FPS{} '.format(args.eval_batch_size * ngpus_per_node / delta_time)
            content = basic + epoch + step + fps + "\n"
            if args.performance:
                file_name = args.fps_acc_dir + '/train_performance_' + str(ngpus_per_node) + 'p_1.txt'
            else:
                file_name = args.fps_acc_dir + '/train_full_' + str(ngpus_per_node) + 'p_1.txt'
            with open(file_name, "a") as writer:
                writer.write(content)


def do_eval(model, task_name, eval_dataloader, loc, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    eval_labels_shuffle = []
    # For eval_dataloader, attribution"shuffle" is set as "True".
    # So predictions and labels are not corresponed with each other in order.
    # Each time the model calculates the predictions(for batch), the corresponding labels must be appended.
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(loc, non_blocking=False) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _ = model(input_ids, segment_ids, input_mask)
        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            eval_labels_shuffle = label_ids
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            eval_labels_shuffle = torch.cat((eval_labels_shuffle.view(-1),label_ids.view(-1)),0)
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(task_name, preds, eval_labels_shuffle)
    result['eval_loss'] = eval_loss

    return result

def list_to_device_id(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    return devices

################## end define classes and functions ##################

def main():

    ################## set args ##################
    parser = argparse.ArgumentParser()

    # 1.file and model
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--fps_acc_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the fps and acc will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--data_url',
                        type=str,
                        default="")

    # 2.training setting
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    parser.add_argument('--amp', default=False, action='store_true', help='use amp to train the model')
    parser.add_argument('--loss_scale', default=128.,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='mode of precision-mixture training')
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=100)
    parser.add_argument('--pred_distill',
                        action='store_true')

    # 3.distributed training
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--addr',
                        default='10.136.181.115',
                        type=str,
                        help='master addr')
    parser.add_argument('--device_list',
                        default='0,1,2,3,4,5,6,7',
                        type=str,
                        help='device id list')
    parser.add_argument('--port',
                        type=str,
                        default='1025')

    # 4.other setting
    parser.add_argument('--performance',
                        action='store_true',
                        help='modify the full training mode to the performance mode, as requested')
    parser.add_argument('--output_txt_step',
                        default=100,
                        type=int,
                        help='Number of steps to export the FPS and accuracy file')
    parser.add_argument('--transfer_learning',
                        action='store_true')
    args = parser.parse_args()
    ################## end set args ##################

    # set logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # set the environment conditions
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    # Prepare devices
    args.device_list = list_to_device_id(args.device_list)
    ngpus_per_node = len(args.device_list)

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ngpus_per_node > 0:
        torch.npu.manual_seed_all(args.seed)

    # distributed training
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,ngpus_per_node))

def main_worker(npu, args, ngpus_per_node):

    ################## initialize distributed training ##################
    dist.init_process_group(backend='hccl', world_size = ngpus_per_node, rank = npu)
    rank = torch.distributed.get_rank()
    args.workers = ngpus_per_node
    device_id = rank + args.device_list[0]
    logger.info("rank ={}".format(rank))
    logger.info("ngpus_per_node={}".format(ngpus_per_node))
    loc = torch.device(f'npu:{rank}')
    torch.npu.set_device(loc)
    ################## end initialize distributed training ##################

    # intermediate distillation default parameters
    processors = {
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "sst-2": Sst2Processor,
    }

    default_params = {
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
    }

    ################## Prepare task settings and SST-2 data ##################
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]

    task_name = args.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size // args.gradient_accumulation_steps // ngpus_per_node)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer)
        train_data, _ = get_tensor_data(train_features)
    ################## end prepare ##################

    ################## set dataloader ##################

        train_sampler = torch.utils.data.DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      shuffle = (train_sampler is None), pin_memory = False, drop_last = True)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
    eval_data, eval_labels = get_tensor_data(eval_features)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size,
                                 shuffle=True, pin_memory=False, drop_last=True)
    ################## end set dataloader ##################

    ################## training&evaluation ##################

    # set the model
    if args.transfer_learning:
        student_model = TinyBertForSequenceClassification_for_finetune.from_pretrained(args.student_model, num_labels=num_labels)
    else:
        student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)
    student_model.to(loc)

    # evaluation
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader, loc, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    #training
    else:
        # parameters and model structure
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("  num_labels = %d", num_labels)
        if args.transfer_learning:
            teacher_model = TinyBertForSequenceClassification_for_finetune.from_pretrained(args.teacher_model,num_labels=num_labels)
        else:
            teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(loc)

        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'

        # optimizer
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        # precision-mixture mode
        if args.amp:
            student_model, optimizer = amp.initialize(student_model, optimizer, combine_grad=True,
                                                      opt_level=args.opt_level, loss_scale=args.loss_scale)

        student_model = DDP(student_model, device_ids=[rank], broadcast_buffers=False)
        if args.amp:
            teacher_model = amp.initialize(teacher_model, combine_grad=True,
                                            opt_level=args.opt_level, loss_scale=args.loss_scale)
        teacher_model = DDP(teacher_model, device_ids=[rank], broadcast_buffers=False)

        # Prepare loss functions
        loss_mse = MSELoss().to(loc)

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train
        global_step = 0
        best_dev_acc = 0.0

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                if args.performance:
                    if (step + 1) == 1000:
                        logger.info("End performance testing. Ready to exit.")
                        sys.exit()
                if rank == 0:
                    start_time = time.time()
                batch = tuple(t.to(loc, non_blocking=False) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask,
                                                                       is_student=True)
                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                # intermediate layer distillation
                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(loc),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(loc),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss
                    # Make sure all the outputs of model must be included in the calculation of loss function
                    # in order that the calculation graph of PyTorch is complete and enclosed.
                    loss = rep_loss + att_loss + 0 * soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    # Make sure all the outputs of model must be included in the calculation of loss function
                    # in order that the calculation graph of PyTorch is complete and enclosed.
                    loss = cls_loss + 0 * loss_mse(student_atts[0], teacher_atts[0])+ 0 * loss_mse(teacher_reps[0], student_reps[0])
                    tr_cls_loss += cls_loss.item()

                if ngpus_per_node > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                if rank == 0:
                    delta_time = time.time() - start_time

            # evaluate
                if (step + 1) % args.eval_step == 0 and rank == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    if args.pred_distill:
                        result = do_eval(student_model, task_name, eval_dataloader, loc, num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    result_to_file(result, args, step, rank, delta_time, epoch_, ngpus_per_node)

                    if not args.pred_distill:
                        save_model = True
                    else:
                        save_model = False

                        if result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        model_name = WEIGHTS_NAME
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        # Test mnli-mm
                        if args.pred_distill and task_name == "mnli":
                            task_name = "mnli-mm"
                            processor = processors[task_name]()
                            if not os.path.exists(args.output_dir + '-MM'):
                                os.makedirs(args.output_dir + '-MM')

                            eval_examples = processor.get_dev_examples(args.data_dir)

                            eval_features = convert_examples_to_features(
                                eval_examples, label_list, args.max_seq_length, tokenizer)
                            eval_data, eval_labels = get_tensor_data(eval_features)

                            logger.info("***** Running mm evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)

                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                         batch_size=args.eval_batch_size)

                            result = do_eval(student_model, task_name, eval_dataloader,
                                             loc, num_labels)
                            result['global_step'] = global_step

                            tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)

                            task_name = 'mnli'
                    student_model.train()


if __name__ == "__main__":
    main()

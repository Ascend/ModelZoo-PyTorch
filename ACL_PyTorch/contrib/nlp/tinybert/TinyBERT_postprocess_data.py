# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""Run TinyBERT on SST-2."""

from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
import csv
import numpy as np
import io
from transformer.tokenization import BertTokenizer
import torch


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
        with io.open(input_file, "r", encoding="utf-8") as f:
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
        """get the augmented examples"""
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


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

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

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def get_label_ids(features):
    """get the label id"""
    return torch.tensor([f.label_id for f in features], dtype=torch.long)


def simple_accuracy(preds, labels):
    """calculate the accuracy"""
    return (preds == labels).mean()


def bin2predlabel(test_num, args):
    """(adapt to benchmark inference)change the bin files into logits"""
    logit1 = []
    logit2 = []
    for i in range(test_num):
        n1, n2 = np.fromfile('{}/Bert_{}_1.bin'.format(args.result_dir, i), dtype='float32')
        logit1.append(n1)
        logit2.append(n2)
    logit = np.concatenate((np.array(logit1).reshape(1, -1), np.array(logit2).reshape(1, -1)), axis = 0)
    pred_label = np.argmax(logit, axis = 0)
    return pred_label


def txt2predlabel(test_num, args):
    """(adapt to msame inference):change the txt files into logits"""
    logit1 = []
    logit2 = []
    for i in range(test_num):
        txtname = "input" + str(i) + "_output_0.txt"
        dir = os.path.join(args.result_dir, txtname)
        with open(dir, "r") as f:
            line = f.readline()
        n1, n2 = [float(i) for i in line.split()]
        logit1.append(n1)
        logit2.append(n2)
    logit = np.concatenate((np.array(logit1).reshape(1, -1), np.array(logit2).reshape(1, -1)), axis = 0)
    pred_label = np.argmax(logit, axis = 0)
    return pred_label


def main():
    """postprocess the data and calculate the accuracy"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--result_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="NPU benchmark infer result path")
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--inference_tool", type = str,
                        help = "inference tool:benchmark or msame")
    args = parser.parse_args()
    test_num = 872
    processor = Sst2Processor()
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    eval_examples = processor.get_dev_examples(args.data_dir)
    label_list = ["0", "1"]
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer,
                                                 output_mode="classification")
    #data processing
    eval_labels = get_label_ids(eval_features).numpy()
    if args.inference_tool == "benchmark":
        pred_labels = bin2predlabel(test_num, args)
    elif args.inference_tool == "msame":
        pred_labels = txt2predlabel(test_num, args)
    result = simple_accuracy(pred_labels, eval_labels)
    print("acc:{}".format(result))

if __name__ == '__main__':
    main()
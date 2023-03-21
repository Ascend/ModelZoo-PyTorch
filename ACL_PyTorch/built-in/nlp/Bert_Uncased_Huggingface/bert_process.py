# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import os

import torch
import datasets
import transformers
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)

from utils_qa import postprocess_qa_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="preprocess and postprocess of bert-base/large-uncased model")

    # Required for preprocess and postprocess
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--process_mode",
        type=str,
        choices=["preprocess", "postprocess"],
        required=True,
    )
    parser.add_argument(
        "--offline_download",
        action="store_true",
        help="If passed, download datasets from local file.",
    )

    # Required for preprocess
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./prep_data",
    )

    # Required for postprocess
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
    )

    # Defaults
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="squad",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    args = parser.parse_args()

    return args


def prepare_validation_features(examples):
    """
    Validation preprocessing
    """
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    examples["question"] = [q.lstrip() for q in examples["question"]] # remove left whitespace
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if args.pad_to_max_length else False,
    )

    # a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # keep the corresponding example_id and store the offset mappings for evaluation
    tokenized_examples["example_id"] = []
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def post_processing_function(examples, features, predictions):
    """
    match the start logits and end logits to answers in the original context.
    """
    args = parse_args()
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    """
    step = 0
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    for output_logit in start_or_end_logits: 
        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]
        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
        step += batch_size
    return logits_concat


def main():
    args = parse_args()

    # Initialize datasets,tokenizer and metric
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if args.offline_download:
        raw_datasets = load_from_disk("./squad/data")
        if args.process_mode == "postprocess":
            metric = load_metric("./squad/metric/squad.py")
    else:
        raw_datasets = load_dataset(args.dataset_name, None)
        if args.process_mode == "postprocess":
            metric = load_metric("squad")

    # Validation Feature Creation
    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=eval_examples.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )

    # DataLoaders Creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Pre-processing
    if args.process_mode == "preprocess":
        for idx, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                for input_name in batch.keys():
                    save_path = os.path.join(args.save_dir, input_name, "{}.npy".format(idx))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, batch[input_name].numpy())

    # Post-processing
    if args.process_mode == "postprocess":
        all_start_logits = []
        all_end_logits = []
        for idx, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                start_logits_path = os.path.join(args.result_dir, "{}_0.npy".format(idx))
                end_logits_path = os.path.join(args.result_dir, "{}_1.npy".format(idx))
                all_start_logits.append(np.load(start_logits_path))
                all_end_logits.append(np.load(end_logits_path))

        max_len = max([x.shape[1] for x in all_start_logits])
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        print(f"Evaluation metrics: {eval_metric}")


if __name__ == "__main__":
    main()
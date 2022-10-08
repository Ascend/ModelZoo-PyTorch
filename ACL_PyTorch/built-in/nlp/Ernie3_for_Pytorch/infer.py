# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
from ast import arg
import os
import time
import sys
from functools import partial
import distutils.util
import numpy as np
import onnxruntime as ort
from multiprocessing import cpu_count

import paddle
from paddle import inference
from paddle.metric import Accuracy
from datasets import load_dataset
from paddlenlp.datasets import load_dataset as ppnlp_load_dataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.data import DataCollatorForTokenClassification, DataCollatorWithPadding
from paddlenlp.transformers import AutoTokenizer
from pyacl.acl_infer import AclNet, init_acl, release_acl
import acl

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='tnews',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()))
    parser.add_argument("--model_name_or_path",
                        default="ernie-3.0-base-zh",
                        type=str,
                        help="The directory or name of model.")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="The path prefix of inference model to be used.")
    parser.add_argument("--device",
                        default="gpu",
                        choices=["gpu", "cpu", "xpu", "npu"],
                        help="Device selected for inference.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size for predict.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
        help=
        "The npu device id")
    parser.add_argument("--perf_warmup_steps",
                        default=20,
                        type=int,
                        help="Warmup steps for performance test.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument("--max_answer_length",
                        default=50,
                        type=int,
                        help="Max answer length for question answering task.")
    parser.add_argument("--shape_file",
                        default="shape_info.txt",
                        type=str,
                        help="Shape info filename.")
    parser.add_argument("--use_trt",
                        action='store_true',
                        help="Whether to use inference engin TensorRT.")
    parser.add_argument("--perf",
                        action='store_true',
                        help="Whether to test performance.")
    parser.add_argument("--collect_shape",
                        action='store_true',
                        help="Whether to collect shape info.")

    parser.add_argument("--precision",
                        default="fp32",
                        choices=["fp32", "fp16", "int8"],
                        help="Precision for inference.")
    parser.add_argument(
        "--num_threads",
        default=cpu_count(),
        type=int,
        help="num_threads for cpu.",
    )
    parser.add_argument(
        "--enable_quantize",
        action='store_true',
        help=
        "Whether to enable quantization for acceleration. Valid for both onnx and dnnl",
    )
    parser.add_argument("--use_onnxruntime",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Use onnxruntime to infer or not.")
    parser.add_argument("--use_pyacl",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Use pyacl to infer or not.")
    parser.add_argument(
        "--debug",
        action='store_true',
        help="With debug it will save graph and model after each pass.")
    parser.add_argument(
        "--provider",
        default='CPUExecutionProvider',
        choices=['CPUExecutionProvider', 'DnnlExecutionProvider'],
        type=str,
        help="Onnx ExecutionProvider with DNNL or without DNNL")

    args = parser.parse_args()
    return args


def convert_example(example,
                    tokenizer,
                    label_list,
                    is_test=False,
                    max_seq_length=512):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = np.array(example["label"], dtype="int64")
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {
            'sentence1': sentence1,
            'sentence2': example['abst'],
            'label': example['label']
        }
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(
            pronoun_idx +
            len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx +
                               len(query))] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example['sentence'] = text
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)

    if not is_test:
        example["labels"] = label
    while len(example['input_ids']) < max_seq_length:
        example['input_ids'].append(0)
        example['token_type_ids'].append(0)
    return example


class Predictor(object):

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        if args.use_pyacl:
            assert args.device == "npu", "Running pyacl on this device is temporarily not supported."
            if args.model_path.count(".om"):
                om_model = args.model_path
            else:
                return
            init_acl(args.device_id)
            predictor = AclNet(model_path=om_model, device_id=args.device_id)
            return cls(predictor, [], [])
        if args.use_onnxruntime:
            assert args.device != "xpu", "Running ONNXRuntime on XPU is temporarily not supported."
            if args.model_path.count(".onnx"):
                onnx_model = args.model_path
            else:
                import paddle2onnx
                onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                    model_file=args.model_path + ".pdmodel",
                    params_file=args.model_path + ".pdiparams",
                    opset_version=13,
                    enable_onnx_checker=True)
            dynamic_quantize_model = onnx_model
            if args.enable_quantize:
                from onnxruntime.quantization import QuantizationMode, quantize_dynamic
                float_onnx_file = "model.onnx"
                with open(float_onnx_file, "wb") as f:
                    f.write(onnx_model)
                dynamic_quantize_model = "dynamic_quantize_model.onnx"
                quantize_dynamic(float_onnx_file, dynamic_quantize_model)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = args.num_threads
            sess_options.inter_op_num_threads = args.num_threads
            executionprovider = args.provider
            print("ExecutionProvider is: ", executionprovider)
            predictor = ort.InferenceSession(dynamic_quantize_model,
                                             sess_options=sess_options,
                                             providers=[executionprovider])
            input_name1 = predictor.get_inputs()[0].name
            input_name2 = predictor.get_inputs()[1].name
            input_handles = [input_name1, input_name2]
            return cls(predictor, input_handles, [])

        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            config.switch_ir_optim(True)
            config.enable_mkldnn()
            if args.enable_quantize:
                config.enable_mkldnn_int8()
            if args.debug:
                config.switch_ir_debug(True)
            config.set_cpu_math_library_num_threads(args.num_threads)
            cls.device = paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_trt:
            precision_map = {
                "int8": inference.PrecisionType.Int8,
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32
            }
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=precision_map[args.precision],
                max_batch_size=args.batch_size,
                min_subgraph_size=5,
                use_static=False,
                use_calib_mode=False)
            print("Enable TensorRT is: {}".format(
                config.tensorrt_engine_enabled()))

            if args.collect_shape:
                config.collect_shape_range_info(args.task_name +
                                                args.shape_file)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(
                    args.task_name + args.shape_file, True)

        config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        predictor = paddle.inference.create_predictor(config)

        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return cls(predictor, input_handles, output_handles)

    def acc_write(self, resstr):
        with open("results.txt", "a") as resfile:
            resfile.write(resstr)
    def pad_batch(self, id1, id2, last_batch, args):
        if id1.shape[0] < args.batch_size:
            last_batch = True
            pad_len = args.batch_size - id1.shape[0]
            pad_data = np.zeros((pad_len, args.max_seq_length), dtype=np.int64)
            id1 = np.concatenate((id1, pad_data), axis=0)
            id2 = np.concatenate((id2, pad_data), axis=0)
        return id1, id2, last_batch

    def set_dynamic_shape(self, max_seq_length, batch_size):
        # The dynamic shape info required by TRT is automatically generated according to max_seq_length and batch_size and stored in shape_info.txt
        min_batch_size, max_batch_size, opt_batch_size = 1, batch_size, batch_size
        min_seq_len, max_seq_len, opt_seq_len = 2, max_seq_length, 32
        batches = [
            [
                np.zeros([min_batch_size, min_seq_len], dtype="int64"),
                np.zeros([min_batch_size, min_seq_len], dtype="int64")
            ],
            [
                np.zeros([max_batch_size, max_seq_len], dtype="int64"),
                np.zeros([max_batch_size, max_seq_len], dtype="int64")
            ],
            [
                np.zeros([opt_batch_size, opt_seq_len], dtype="int64"),
                np.zeros([opt_batch_size, opt_seq_len], dtype="int64")
            ],
        ]
        for batch in batches:
            self.predict_batch(batch)
        print(
            "Set dynamic shape finished, please close set_dynamic_shape and restart."
        )
        exit(0)

    def predict_batch(self, data):
        if len(self.output_handles) == 0 and len(self.input_handles) == 0:
            result, t = self.predictor(data)
            return result
        if len(self.output_handles) == 0:
            input_dict = {}
            for input_field, input_handle in zip(data, self.input_handles):
                input_dict[input_handle] = input_field
            result = self.predictor.run(None, input_dict)
            return result

        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self,
                dataset,
                tokenizer,
                batchify_fn,
                args,
                dev_example=None,
                dev_ds_ori=None):
        if args.collect_shape:
            self.set_dynamic_shape(args.max_seq_length, args.batch_size)
        if args.task_name == "cmrc2018":
            dataset_removed = dataset.remove_columns(
                ["offset_mapping", "attention_mask", "example_id"])
            sample_num = len(dataset)
            batches = []
            for i in range(0, sample_num, args.batch_size):
                batch_size = min(args.batch_size, sample_num - i)
                batch = [dataset_removed[i + j] for j in range(batch_size)]
                batches.append(batch)
        else:
            sample_num = len(dataset)
            batches = []
            for i in range(0, sample_num, args.batch_size):
                batch_size = min(args.batch_size, sample_num - i)
                batch = [dataset[i + j] for j in range(batch_size)]
                batches.append(batch)

        if args.task_name == "msra_ner":
            metric = ChunkEvaluator(label_list=args.label_list)
            metric.reset()
            all_predictions = []
            batch_num = len(dataset['input_ids'])
            for batch in batches:
                batch = batchify_fn(batch)
                input_ids, segment_ids = batch["input_ids"].numpy(
                ), batch["token_type_ids"].numpy()
                last_batch_num = input_ids.shape[0]
                last_batch = False
                input_ids, segment_ids, last_batch = self.pad_batch(input_ids, segment_ids, last_batch, args)
                output = self.predict_batch([segment_ids, input_ids])[0]
                if last_batch:
                    output = output[:, last_batch_num, ...]
                preds = np.argmax(output, axis=2)
                all_predictions.append(preds.tolist())
                num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
                    batch["seq_len"], paddle.to_tensor(preds),
                    batch["labels"])
                metric.update(num_infer_chunks.numpy(),
                                num_label_chunks.numpy(),
                                num_correct_chunks.numpy())
            res = metric.accumulate()
            print("task name: %s, (precision, recall, f1): %s, " %
                    (args.task_name, res))
            resstr = "precision: {}\n".format(res[0])
            self.acc_write(resstr)
            resstr = "recall: {}\n".format(res[1])
            self.acc_write(resstr)
            resstr = "f1: {}\n".format(res[2])
            self.acc_write(resstr)
        elif args.task_name == "cmrc2018":
            all_start_logits = []
            all_end_logits = []
            for batch in batches:
                batch = batchify_fn(batch)
                input_ids, segment_ids = batch["input_ids"].numpy(
                ), batch["token_type_ids"].numpy()
                last_batch_num = input_ids.shape[0]
                last_batch = False
                input_ids, segment_ids, last_batch = self.pad_batch(input_ids, segment_ids, last_batch, args)
                start_logits, end_logits = self.predict_batch(
                    [segment_ids, input_ids])
                if last_batch:
                    start_logits = start_logits[:last_batch_num, :]
                    end_logits = end_logits[:last_batch_num, :]
                for idx in range(start_logits.shape[0]):
                    if len(all_start_logits) % 1000 == 0 and len(
                            all_start_logits):
                        print("Processing example: %d" %
                                len(all_start_logits))
                    all_start_logits.append(start_logits[idx])
                    all_end_logits.append(end_logits[idx])
            all_predictions, _, _ = compute_prediction(
                dev_example, dataset, (all_start_logits, all_end_logits),
                False, args.n_best_size, args.max_answer_length)
            res = squad_evaluate(
                examples=[raw_data for raw_data in dev_example],
                preds=all_predictions,
                is_whitespace_splited=False)
            print("task name: %s, EM: %s, F1: %s" %
                    (args.task_name, res['exact'], res['f1']))
            resstr = "EM: {}\n".format(res['exact'])
            self.acc_write(resstr)
            resstr = "F1: {}\n".format(res['f1'])
            self.acc_write(resstr)
            return all_predictions
        else:
            all_predictions = []
            metric = METRIC_CLASSES[args.task_name]()
            metric.reset()
            for i, batch in enumerate(batches):
                batch = batchify_fn(batch)
                last_batch = False
                input_ids = batch["input_ids"].numpy()
                token_type_ids = batch["token_type_ids"].numpy()
                last_batch_num = input_ids.shape[0]
                input_ids, token_type_ids, last_batch = self.pad_batch(input_ids, token_type_ids, last_batch, args)
                output = self.predict_batch([
                    input_ids,
                    token_type_ids
                ])[0]
                if last_batch:
                    output = output[:last_batch_num, :]
                preds = np.argmax(output, axis=1)
                all_predictions.append(preds.tolist())
                correct = metric.compute(paddle.to_tensor(output),
                                            batch["labels"])
                metric.update(correct)
            res = metric.accumulate()

            print("task name: %s, acc: %s, " % (args.task_name, res))
            resstr = "acc: {}\n".format(res)
            self.acc_write(resstr)
            return all_predictions


def tokenize_and_align_labels(example,
                              tokenizer,
                              no_entity_id,
                              max_seq_len=512):
    if example['tokens'] == []:
        tokenized_input = {
            'labels': [],
            'input_ids': [],
            'token_type_ids': [],
            'seq_len': 0,
            'length': 0,
        }
        return tokenized_input
    tokenized_input = tokenizer(
        example['tokens'],
        max_seq_len=max_seq_len,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        return_length=True)
    label_ids = example['ner_tags']
    if len(tokenized_input['input_ids']) - 2 < len(label_ids):
        label_ids = label_ids[:len(tokenized_input['input_ids']) - 2]
    label_ids = [no_entity_id] + label_ids + [no_entity_id]

    label_ids += [no_entity_id
                  ] * (len(tokenized_input['input_ids']) - len(label_ids))
    tokenized_input["labels"] = label_ids
    while len(tokenized_input['input_ids']) < max_seq_len:
        tokenized_input['input_ids'].append(0)
        tokenized_input['token_type_ids'].append(0)
        tokenized_input['labels'].append(0)
    tokenized_input['length'] = max_seq_len
    tokenized_input['seq_len'] = max_seq_len
    return tokenized_input


def prepare_validation_features(examples, tokenizer, doc_stride,
                                max_seq_length):
    contexts = examples['context']
    questions = examples['question']

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=doc_stride,
                                   max_seq_len=max_seq_length,
                                   return_attention_mask=True)

    sample_mapping = tokenized_examples.pop("overflow_to_sample")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples['token_type_ids'][i]
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index
             and k != len(sequence_ids) - 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()

    predictor = Predictor.create_predictor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.task_name == "msra_ner":

        def ner_trans_fn(example,
                         tokenizer,
                         max_seq_length=128,
                         no_entity_id=0):
            return tokenize_and_align_labels(example,
                                             tokenizer=tokenizer,
                                             no_entity_id=no_entity_id,
                                             max_seq_len=max_seq_length)

        trans_fn = partial(ner_trans_fn,
                           tokenizer=tokenizer,
                           max_seq_length=args.max_seq_length)
        dev_ds = load_dataset("msra_ner", split="test")
        label_list = dev_ds.features['ner_tags'].feature.names
        args.label_list = label_list

        column_names = dev_ds.column_names
        dev_ds = dev_ds.map(trans_fn, remove_columns=column_names)
        batchify_fn = DataCollatorForTokenClassification(tokenizer)
        outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)
    elif args.task_name == "cmrc2018":
        dev_example = load_dataset("cmrc2018", split="validation")
        column_names = dev_example.column_names
        dev_ds = dev_example.map(
            partial(prepare_validation_features,
                    tokenizer=tokenizer,
                    doc_stride=128,
                    max_seq_length=args.max_seq_length),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on validation dataset",
        )

        batchify_fn = DataCollatorWithPadding(tokenizer)
        outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args,
                                    dev_example)
    else:
        dev_ds = ppnlp_load_dataset('clue', args.task_name, splits='dev')

        trans_func = partial(convert_example,
                             label_list=dev_ds.label_list,
                             tokenizer=tokenizer,
                             max_seq_length=args.max_seq_length,
                             is_test=False)
        dev_ds = dev_ds.map(trans_func, lazy=False)
        batchify_fn = DataCollatorWithPadding(tokenizer)

        outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()

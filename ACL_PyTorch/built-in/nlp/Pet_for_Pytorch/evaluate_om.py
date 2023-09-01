# Copyright 2023 Huawei Technologies Co., Ltd
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

from functools import partial

from dataclasses import dataclass, field
import numpy as np
import paddle
from data import load_fewclue_dataset
from paddle.metric import Accuracy
from utils import load_prompt_arguments
from paddlenlp.prompt import (
    ManualTemplate,
    MaskedLMVerbalizer,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger

from ais_bench.infer.interface import InferSession

@dataclass
class DataArguments:
    task_name: str = field(default="eprstmt", \
                           metadata={"help": "The task name in FewCLUE."})
    split_id: str = field(default="0", \
                          metadata={"help": "The split id of datasets."})
    device_id: int = field(default=0, \
            metadata={"help": "The id of npu device that you want to use."})
    om_path: str = field(default="pet.om", \
                         metadata={"help": "The path of om model."})
    prompt_path: str = field(default="prompt/eprstmt.json", \
            metadata={"help": "Path to the defined prompts."})
    prompt_index: int = field(default=0, \
            metadata={"help": "The index of defined prompt for training."})
    augment_type: str = field(default=None, \
            metadata={"help": "The strategy used for data augmentation."})
    num_augment: str = field(default=5, metadata={"help":"Number of augmented\
             data per example, which works when `augment_type` is set."})
    word_augment_percent: str = field(default=0.1, \
            metadata={"help": "Percentage of augmented words in sequences."})
    augment_method: str = field(default="mlm", \
            metadata={"help": "Strategy used for `insert` and `subsitute`."})
    pseudo_data_path: str = field(default=None, \
                    metadata={"help": "Path to data with pseudo labels."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata=\
                    {"help": "model name or the path to local model."})
    dropout: float = field(default=0.1, \
                metadata={"help": "The dropout used for pretrained model."})


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments,\
                    DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args = load_prompt_arguments(data_args)
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )

    # Define template for preprocess and verbalizer for postprocess.
    template = ManualTemplate(data_args.prompt,\
            tokenizer, training_args.max_seq_length)
    logger.info("Using template: {}".format(template.prompt))

    verbalizer = MaskedLMVerbalizer(data_args.label_words, tokenizer)
    logger.info("Using verbalizer: {}".format(data_args.label_words))

    # Load datasets.
    data_ds, label_list = load_fewclue_dataset(data_args,\
        verbalizer=verbalizer, example_keys=template.example_keys)
    train_ds, dev_ds, public_test_ds, _, _ = data_ds
    dev_labels, test_labels = label_list

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, verbalizer, freeze_plm=training_args.freeze_plm,\
              freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds, labels, verbalizer):
        metric = Accuracy()
        predictions = paddle.to_tensor(eval_preds)
        predictions = verbalizer.aggregate_multiple_mask(predictions)
        correct = metric.compute(predictions, paddle.to_tensor(labels))
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    # Initialize the trainer.
    dev_compute_metrics = partial(compute_metrics,\
                                labels=dev_labels, verbalizer=verbalizer)
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=dev_compute_metrics,
    )

    # Om evaluate
    testdata = trainer.get_test_dataloader(public_test_ds)
    predictor = InferSession(data_args.device_id, data_args.om_path)
    record = None
    for inputs in testdata:
        pad_batch = training_args.per_device_eval_batch_size \
                    - inputs['input_ids'].shape[0]
        pad_size = training_args.max_seq_length - inputs['input_ids'].shape[1]
        inputs['input_ids'] \
            = np.pad(inputs['input_ids'].cpu().numpy().astype('int64'),\
                                    [(0, pad_batch), (0, pad_size)])
        inputs['token_type_ids'] \
            = np.pad(inputs['token_type_ids'].cpu().numpy().astype('int64'),\
                                        [(0, pad_batch), (0, pad_size)])
        inputs['position_ids'] \
            = np.pad(inputs['position_ids'].cpu().numpy().astype('int64'), \
                                        [(0, pad_batch), (0, pad_size)])
        inputs['attention_mask'] = paddle.unsqueeze(
            (paddle.to_tensor(inputs['input_ids']) == 0).astype('float32')\
                  * -6e5, axis=[1, 2]
        )
        model_input = [
            inputs['input_ids'],
            inputs['token_type_ids'],
            inputs['position_ids'],
            inputs['attention_mask'].cpu().numpy(),
        ]
        results = predictor.infer(model_input)
        results = np.array(results[0])
        results = np.expand_dims(results[:, -1, :], 1)
        if pad_batch > 0:
            results = results[: training_args.per_device_eval_batch_size\
                               - pad_batch, :, :]
        record = results if record is None else \
            np.concatenate((record, results), axis=0)

    test_compute_metrics = partial(compute_metrics, \
                                   labels=test_labels, verbalizer=verbalizer)
    res = test_compute_metrics(record)
    print(res)

if __name__ == "__main__":
    main()

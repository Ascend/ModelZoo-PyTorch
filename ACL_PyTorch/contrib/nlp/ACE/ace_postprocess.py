# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================

import argparse
import glob
import os
import sys
import time
import torch
import numpy as np

sys.path.append('./ACE/')
from flair.utils.from_params import Params
from flair.config_parser import ConfigParser, log
from flair.trainers import ReinforcementTrainer
from flair.custom_data_loader import ColumnDataLoader
from flair.training_utils import Metric, Result
# ============================================================================
# Functions
# ============================================================================
longest_token_sequence_in_batch = 124


def get_trainer(config_path):
    config = Params.from_file(config_path)
    config = ConfigParser(config, all=False, zero_shot=False, other_shot=False, predict=False)

    student = config.create_student(nocrf=False)
    corpus = config.corpus

    trainer = ReinforcementTrainer(student, None, corpus, config=config.config, **config.config["ReinforcementTrainer"],
                                   is_test=True)
    base_path = config.get_target_path

    trainer.embeddings_storage_mode = 'cpu'

    trainer.model.eval()
    trainer.model.to('cpu')

    trainer.model = trainer.model.load(base_path / "best-model.pt", device='cpu')
    training_state = torch.load(base_path / 'training_state.pt', map_location=torch.device('cpu'))
    trainer.best_action = training_state['best_action']
    trainer.model.selection = trainer.best_action

    for embedding in trainer.model.embeddings.embeddings:
        # manually fix the bug for the tokenizer becoming None
        if hasattr(embedding, 'tokenizer') and embedding.tokenizer is None:
            from transformers import AutoTokenizer
            name = embedding.name
            if '_v2doc' in name:
                name = name.replace('_v2doc', '')
            if '_extdoc' in name:
                name = name.replace('_extdoc', '')
            embedding.tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=True)
        if hasattr(embedding, 'model') and hasattr(embedding.model, 'encoder') and not hasattr(
                embedding.model.encoder,
                'config'):
            embedding.model.encoder.config = embedding.model.config
    return trainer


def get_loader(trainer):
    loader = ColumnDataLoader(list(trainer.corpus.test), int(args.batch_size),
                              use_bert=trainer.use_bert,
                              tokenizer=trainer.bert_tokenizer, model=trainer.model,
                              sentence_level_batch=trainer.sentence_level_batch, sort_data=True)

    loader.assign_tags(trainer.model.tag_type, trainer.model.tag_dictionary)
    return loader


def run_postprocess(args):
    bin_file_num = len(glob.glob(os.path.join(args.bin_file_path, "*.bin")))
    if os.path.exists(args.res_file_path) == False:
        os.mkdir(args.res_file_path)
    res_file_path = os.path.join(args.res_file_path, "accuracy.txt")
    if bin_file_num >= 1:
        log.info(f"post process {bin_file_num} file!")
    else:
        log.info("please input right path!")
        exit()
    trainer = get_trainer(args.config)

    loader = get_loader(trainer)

    log.info("---------------ace model starts postprocessing-----------------")
    metric = Metric("Evaluation")

    for i in range(bin_file_num):
        file_path = os.path.join(args.bin_file_path, "in_" + str(i) + "_0.bin")
        if not os.path.exists(file_path):
            continue
        features = np.fromfile(file_path, dtype=np.float32).reshape(
            [int(args.batch_size), longest_token_sequence_in_batch, 20])
        if i == len(loader)-1:
            features = features[:len(loader[i]), :, :]
        tags, _ = trainer.model._obtain_labels(torch.tensor(features), loader[i])

        for (sentence, sent_tags) in zip(loader[i], tags):
            for (token, tag) in zip(sentence.tokens, sent_tags):
                token: Token = token
                token.add_tag_label("predicted", tag)

        for sentence in loader[i]:
            # make list of gold tags
            gold_tags = [
                (tag.tag, str(tag)) for tag in sentence.get_spans("ner")
            ]
            # make list of predicted tags
            predicted_tags = [
                (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
            ]

            # check for true positives, false positives and false negatives
            for tag, prediction in predicted_tags:
                if (tag, prediction) in gold_tags:
                    metric.add_tp(tag)
                else:
                    metric.add_fp(tag)

            for tag, gold in gold_tags:
                if (tag, gold) not in predicted_tags:
                    metric.add_fn(tag)
                else:
                    metric.add_tn(tag)

    detailed_result = (
        f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
        f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
    )
    for class_name in metric.get_classes():
        detailed_result += (
            f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
            f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
            f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
            f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
            f"{metric.f_score(class_name):.4f}"
        )
    result = Result(
        main_score=metric.micro_avg_f_score(),
        log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
        log_header="PRECISION\tRECALL\tF1",
        detailed_results=detailed_result,
    )
    log.info(result.log_line)
    log.info(result.detailed_results)
    with open(res_file_path, "w", encoding="utf-8") as file:
        file.write(result.detailed_results)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/')
    parser.add_argument('--bin_file_path', default='./result/')
    parser.add_argument('--res_file_path', default='./res_data/')
    parser.add_argument('--batch_size', default=1)
    args = parser.parse_args()

    start = time.time()
    run_postprocess(args)
    elapsed = (time.time() - start)
    print("Time used:", elapsed, "s")

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

"""
Official evaluation script for ReCoRD v1.0.
(Some functions are adopted from the SQuAD evaluation script.)
"""

from __future__ import print_function
from collections import Counter
import string
import re
from tasks.data_utils import InputExample
from typing import List
import functools
from collections import defaultdict
import unidecode


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return unidecode.unidecode(text.lower())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return 0.0
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def qa_evaluate(predictions, labels, examples: List[InputExample], metric):
    assert len(examples) == len(predictions)
    score = 0.0
    for example, prediction in zip(examples, predictions):
        ground_truths = example.meta["answers"]
        prediction = example.meta["candidates"][prediction]
        if ground_truths:
            score += metric_max_over_ground_truths(metric, prediction, ground_truths)
    score = 100.0 * score / len(predictions)
    return score


def squad_evaluate(predictions, labels, examples, metric):
    assert len(examples) == len(predictions)
    score = 0.0
    idx2predictions = {}
    idx2ground_truths = {}
    for example, prediction in zip(examples, predictions):
        idx = example.idx
        if idx not in idx2predictions:
            idx2predictions[idx] = []
            idx2ground_truths[idx] = example.meta["answers"]
        idx2predictions[idx].append(prediction)
    # assert len(predictions) == len(idx2predictions)
    for idx, predictions in idx2predictions.items():
        prediction = 'N/A'
        for i in range(len(predictions)):
            prediction = predictions[i]
            if prediction.lower().replace(' ', '') == 'n/a':
                prediction = 'N/A'
            else:
                break
        ground_truths = idx2ground_truths[idx]
        if len(ground_truths) == 1 and ground_truths[0] == 'N/A':
            score += (prediction == 'N/A')
        else:
            score += metric_max_over_ground_truths(metric, prediction, ground_truths)
    score = 100.0 * score / len(idx2predictions)
    return score


def multirc_em(predictions, labels, examples: List[InputExample]):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    question_ids = [example.meta["question_idx"] for example in examples]
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, labels))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)
    return em


qa_exact_match = functools.partial(qa_evaluate, metric=exact_match_score)
qa_f1 = functools.partial(qa_evaluate, metric=f1_score)

squad_exact_match = functools.partial(squad_evaluate, metric=exact_match_score)
squad_f1 = functools.partial(squad_evaluate, metric=f1_score)

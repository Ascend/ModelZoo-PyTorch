# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2021 Huawei Technologies Co., Ltd
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

import math
import logging

import numpy as np

import torch

import pdb


class EvalMetric(object):

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.reset()

    def update(self, preds, labels, losses):
        raise NotImplementedError()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self, prefix=''):
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        name = [prefix + x for x in name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def check_label_shapes(self, preds, labels):
        # raise if the shape is inconsistent
        if (type(labels) is list) and (type(preds) is list):
            label_shape, pred_shape = len(labels), len(preds)
        else:
            label_shape, pred_shape = labels.shape[0], preds.shape[0]

        if label_shape != pred_shape:
            raise NotImplementedError("")


class MetricList(EvalMetric):
    """Handle multiple evaluation metric
    """
    def __init__(self, *args, name="metric_list"):
        assert all([issubclass(type(x), EvalMetric) for x in args]), \
            "MetricList input is illegal: {}".format(args)
        self.metrics = [metric for metric in args]
        super(MetricList, self).__init__(name=name)

    def update(self, preds, labels, losses=None):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels
        losses = [losses] if type(losses) is not list else losses

        for metric in self.metrics:
            metric.update(preds, labels, losses)

    def reset(self):
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                metric.reset()
        else:
            logging.warning("No metric defined.")

    def get(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get())
        return ouputs

    def get_name_value(self, **kwargs):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get_name_value(**kwargs))
        return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
    """Computes accuracy classification score.
    """
    def __init__(self, name='accuracy', topk=1):
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def update(self, preds, labels, losses):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels

        self.check_label_shapes(preds, labels)
        for pred, label in zip(preds, labels):
            assert self.topk <= pred.shape[1], \
                "topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
            _, pred_topk = pred.topk(self.topk, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))
            # print('========================================TODO')
            # print(correct)
            # print(correct.shape)
            # print(self.sum_metric)
            # print(correct.reshape(-1))
            # print(float(correct.view(-1).float().sum(0, keepdim=True).numpy()))
            # self.sum_metric += float(correct.view(-1).float().sum(0, keepdim=True).numpy())
            self.sum_metric += float(correct.contiguous().view(-1).float().sum(0, keepdim=True).numpy()) # TODO （view 需要数据连续存储，所以先调用 contiguous ）
            self.num_inst += label.shape[0]


class Loss(EvalMetric):
    """Dummy metric for directly printing loss.
    """
    def __init__(self, name='loss'):
        super(Loss, self).__init__(name)

    def update(self, preds, labels, losses):
        assert losses is not None, "Loss undefined."
        for loss in losses:
            self.sum_metric += float(loss.numpy().sum())
            self.num_inst += loss.numpy().size


if __name__ == "__main__":
    import torch

    # Test Accuracy
    predicts = [torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]]))]
    labels   = [torch.from_numpy(np.array([   0,            1,          1 ]))]
    losses   = [torch.from_numpy(np.array([   0.3,       0.4,       0.5   ]))]

    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("input pred:  {}".format(predicts))
    logging.debug("input label: {}".format(labels))
    logging.debug("input loss: {}".format(labels))

    acc = Accuracy()

    acc.update(preds=predicts, labels=labels, losses=losses)

    logging.info(acc.get())

    # Test MetricList
    metrics = MetricList(Loss(name="ce-loss"),
                         Accuracy(topk=1, name="acc-top1"),
                         Accuracy(topk=2, name="acc-top2"),
                         )
    metrics.update(preds=predicts, labels=labels, losses=losses)

    logging.info("------------")
    logging.info(metrics.get_name_value(prefix='ts-'))
    logging.info("------ with prefix ------")
    logging.info(acc.get_name_value(prefix='ts-'))

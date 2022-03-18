#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""Copyright 漏 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from metric.diversity_at_k import DiversityAtK
from metric.hit_ratio_at_k import HitRatioAtK
from metric.NDCG_at_k import NDCGAtK
from metric.precision_at_k import PrecisionAtK
import numpy as np
import math
import torch
import pytest

# Packages needed to run test:
# abc
# math
# numpy
# bottleneck
# scipy
# pytest

# Variables
k = 2
y_pred = torch.from_numpy(np.ones((2, 3)))
y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
revenue = np.ones(3)
diversity_vector = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]])


# Helper functions
# Creates a list of metric objects. Add new classes as they are created.
def create_metric_list(k, revenue):
    return [RecallAtK(k), RevenueAtK(k, revenue),
            DiversityAtK(k, diversity_vector), HitRatioAtK(k), NDCGAtK(k),
            PrecisionAtK(k)]


# Creates a list of metric names. Add new classes as they are created.
def create_metric_name_list(k):
    return [s + str(k) for s in ['Recall at ', 'Revenue at ', 'Diversity at ',
            'Hit Ratio at ', 'NDCG at ', 'Precision at ']]


# Metric objects cannot be created with float k or k < 1
def test_metric_check_init():
    classes = [RecallAtK, HitRatioAtK, NDCGAtK, PrecisionAtK]
    # k not set and k is not an int
    for k_test in [None, 2.5]:
        for child in classes:
            with pytest.raises(TypeError):
                child(k_test)
        with pytest.raises(TypeError):
            RevenueAtK(k_test, revenue)
        with pytest.raises(TypeError):
            DiversityAtK(k_test, diversity_vector)
    # k is < 1
    for child in classes:
        with pytest.raises(ValueError):
            child(0)
    with pytest.raises(ValueError):
        RevenueAtK(0, revenue)
    with pytest.raises(ValueError):
        DiversityAtK(0, diversity_vector)


# Name method check.
def test_metric_names():
    metrics = create_metric_list(k, revenue)
    metric_names = create_metric_name_list(k)
    for i, metric in enumerate(metrics):
        assert(metric.get_name() == metric_names[i])


# Evaluate method raises TypeError if y_pred is None.
def test_metric_evaluate_y_pred_none():
    metrics = create_metric_list(k, revenue)
    for metric in metrics:
        with pytest.raises(TypeError):
            metric.evaluate(y_true, None)


# Evaluate method raises TypeError if y_true is None.
def test_metric_evaluate_y_true_none():
    metrics = create_metric_list(k, revenue)
    for metric in metrics:
        with pytest.raises(TypeError):
            metric.evaluate(None, y_pred)


# Evaluate method raises TypeError if y_pred and y_true are None.
def test_metric_evaluate_args_none():
    metrics = create_metric_list(k, revenue)
    for metric in metrics:
        with pytest.raises(TypeError):
            metric.evaluate(None, None)


# Evaluate method raises ValueError if y_pred is not 2D.
def test_metric_evaluate_y_pred_not_2D():
    metrics = create_metric_list(k, revenue)
    y_pred = torch.from_numpy(np.empty((2, 3, 3)))
    for metric in metrics:
        with pytest.raises(ValueError):
            metric.evaluate(y_true, y_pred)
    y_pred = torch.from_numpy(np.empty(2))
    for metric in metrics:
        with pytest.raises(ValueError):
            metric.evaluate(y_true, y_pred)


# Evaluate method raises ValueError if y_true is not 2D.
def test_metric_evaluate_y_true_not_2D():
    metrics = create_metric_list(k, revenue)
    y_true = torch.from_numpy(np.empty((2, 3, 3)))
    for metric in metrics:
        with pytest.raises(ValueError):
            metric.evaluate(y_true, y_pred)
    y_true = torch.from_numpy(np.empty(2))
    for metric in metrics:
        with pytest.raises(ValueError):
            metric.evaluate(y_true, y_pred)


# Evaluate method raises ValueError if y_pred and y_true are 2D but
# not the same shape on those 2 axis.
def test_metric_evaluate_y_pred_y_true_diff_dims():
    metrics = create_metric_list(k, revenue)
    y_true = torch.from_numpy(np.empty((2, 3)))
    y_pred = torch.from_numpy(np.empty((3, 2)))
    for metric in metrics:
        with pytest.raises(ValueError):
            metric.evaluate(y_true, y_pred)


# Evaluate method should give zero if all y_pred are 0 regardless of y_true
def test_metric_evaluate_y_pred_zeros():
    metrics = create_metric_list(k, np.ones(3))
    y_pred = torch.from_numpy(np.zeros((2, 3)))
    for metric in metrics:
        assert(metric.evaluate(y_true, y_pred) == 0.0)


# Class specific tests:
# RevenueAtK specific test for revenue argument and dimensions
def test_revenue_evaluate_revenue_dimensions():
    with pytest.raises(TypeError):
        RevenueAtK(k, None)
    with pytest.raises(TypeError):
        RevenueAtK(k, [1, 2, 3])
    with pytest.raises(TypeError):
        RevenueAtK(k, np.ones((2, 3)))
    with pytest.raises(TypeError):
        RevenueAtK(k, np.array(['one', 2]))
    with pytest.raises(ValueError):
        RevenueAtK(k, np.array([]))
    # Dimension check
    revenue = RevenueAtK(k, np.ones(4))
    with pytest.raises(ValueError):
        revenue.evaluate(y_true, y_pred)


# DiversityAtK specific test for dimensions
def test_diversity_evaluate_diversity_dimensions():
    with pytest.raises(TypeError):
        DiversityAtK(k, None)
    with pytest.raises(TypeError):
        DiversityAtK(k, [1, 2, 3])
    with pytest.raises(ValueError):
        DiversityAtK(k, np.array([]))
    # Dimensio check
    diversity = DiversityAtK(k, np.ones(4))
    with pytest.raises(ValueError):
        diversity.evaluate(y_true, y_pred)


# HitRatioAtK specific test for y_true input
def test_hit_ratio_evaluate_y_true():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    hr = HitRatioAtK(k)
    with pytest.raises(ValueError):
        hr.evaluate(y_true, y_pred)


# NDCGAtK specific test for y_true input
def NDCG_evaluate_y_true():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    ndcg = NDCGAtK(k)
    with pytest.raises(ValueError):
        ndcg.evaluate(y_true, y_pred)


# RecallAtK evaluate output check
def test_recall_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    recall = RecallAtK(k)
    assert(recall.evaluate(y_true, y_pred) == 1.0)
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[1, 0, 1], [1, 1, 0]]))
    assert(recall.evaluate(y_true, y_pred) == 0.5)
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(recall.evaluate(y_true, y_pred) == 0.0)
    y_pred = torch.from_numpy(np.array([[0.1, 0, 0.2], [0, 0.1, 0.7]]))
    assert(recall.evaluate(y_true, y_pred) == 1)


# RevenueAtK evaluate output check
def test_revenue_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    revenue_vector = np.ones(3)
    revenue = RevenueAtK(k, revenue_vector)
    assert(revenue.evaluate(y_true, y_pred) == 2.0)
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[1, 0, 1], [1, 1, 0]]))
    assert(revenue.evaluate(y_true, y_pred) == 1.0)
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(revenue.evaluate(y_true, y_pred) == 0.0)


# DiversityAtK evaluate output check
def test_diversity_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    diversity_vector = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]])
    diversity = DiversityAtK(k, diversity_vector)
    assert(diversity.evaluate(y_true, y_pred) == 0.5)
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[1, 0, 1], [1, 1, 0]]))
    assert(diversity.evaluate(y_true, y_pred) == 0.375)
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(diversity.evaluate(y_true, y_pred) == 0.375)


# HitRatioAtK evaluate output check
def test_hit_ratio_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[0, 0, 1], [1, 0, 0]]))
    hr = HitRatioAtK(k)
    assert(hr.evaluate(y_true, y_pred) == 1.0)
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(hr.evaluate(y_true, y_pred) == 0)


# NDCGAtK evaluate output check
def test_NDCG_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    ndcg = NDCGAtK(k)
    assert(ndcg.evaluate(y_true, y_pred) == 1)
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[0, 0, 1], [1, 0, 0]]))
    assert(ndcg.evaluate(y_true, y_pred) == (1 / math.log(3, 2)))
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(ndcg.evaluate(y_true, y_pred) == 0)


# PrecisionAtK evaluate output check
def test_precision_evaluate_correct_cases():
    y_pred = torch.from_numpy(np.ones((2, 3)))
    y_true = torch.from_numpy(np.ones((2, 3)))
    precision = PrecisionAtK(k)
    assert(precision.evaluate(y_true, y_pred) == 1.0)
    y_pred = torch.from_numpy(np.array([[0.1, 0.4, 0.2], [0.5, 0.1, 0.7]]))
    y_true = torch.from_numpy(np.array([[1, 0, 1], [1, 1, 0]]))
    assert(precision.evaluate(y_true, y_pred) == 0.5)
    y_true = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]]))
    assert(precision.evaluate(y_true, y_pred) == 0.0)
    y_pred = torch.from_numpy(np.array([[0.1, 0, 0.2], [0, 0.1, 0.7]]))
    assert(precision.evaluate(y_true, y_pred) == 0.5)

# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import numpy


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == numpy.max(GT)
    corr = numpy.sum(SR == GT)
    tensor_size = SR.shape[0] * SR.shape[1] * SR.shape[2] * SR.shape[3]
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == numpy.max(GT)
    TP = SR & GT
    FN = (~SR) & GT

    SE = float(numpy.sum(TP)) / (float(numpy.sum(TP) + numpy.sum(FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == numpy.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = (~SR) & (~GT)
    FP = SR & (~GT)

    SP = float(numpy.sum(TN)) / (float(numpy.sum(TN) + numpy.sum(FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    # print(SR.type)
    SR = SR > threshold
    GT = GT == numpy.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = SR & GT
    FP = SR & (~GT)

    PC = float(numpy.sum(TP)) / (float(numpy.sum(TP) + numpy.sum(FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == numpy.max(GT)

    Inter = numpy.sum((SR & GT))
    Union = numpy.sum((SR | GT))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == numpy.max(GT)

    Inter = numpy.sum((SR & GT))
    DC = float(2 * Inter) / (float(numpy.sum(SR) + numpy.sum(GT)) + 1e-6)

    return DC

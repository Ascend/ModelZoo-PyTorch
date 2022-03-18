# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================import numpy as np


def bbox2result(bboxes, labels, num_classes, thr=0.01):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 4)
        labels (Tensor): shape (n, #num_classes)
        num_classes (int): class number, including background class
        thr (float): The score threshold used when converting predictions to
            detection results
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))

    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    # We only handle multilabel now
    assert labels.shape[-1] > 1

    scores = labels  # rename for clarification
    thr = (thr, ) * num_classes if isinstance(thr, float) else thr
    assert scores.shape[1] == num_classes
    assert len(thr) == num_classes

    result = []
    for i in range(num_classes - 1):
        where = scores[:, i + 1] > thr[i + 1]
        result.append(
            np.concatenate((bboxes[where, :4], scores[where, i + 1:i + 2]),
                           axis=1))
    return result

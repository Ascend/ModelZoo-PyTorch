# Copyright 2021 Huawei Technologies Co., Ltd
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
import mmcv
import argparse
import os
import numpy as np

def postProcesss(result_path):
    mulu_bin_path = os.listdir(result_path)[0]
    mulu_result_path = os.path.join(result_path, mulu_bin_path)
    bin_list = os.listdir(mulu_result_path)
    outputs = []
    labels = []
    for i in range(len(bin_list)):
        bin_path = os.path.join(mulu_result_path, bin_list[i])
        label = bin_list[i].split('_')[1:-1]

        if len(label)==1:
            output = np.loadtxt(bin_path)
            res_out = np.mean(output,axis=0)
            outputs.append(res_out)
            labels.append(int(label[0]))
        else:
            res_out = []
            output = np.loadtxt(bin_path)
            for i in range(0,output.shape[0],3):
                tmp = np.mean(output[i:i+3],axis=0)
                res_out.append(tmp)
            outputs.extend(res_out)
            labels.extend(int(i) for i in label)

    return outputs,labels


def mean_class_accuracy(scores, labels):

    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc

def top_k_accuracy(scores, labels, topk=(1, )):
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

def confusion_matrix(y_pred, y_real, normalize=None):

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='postprocess of r2plus1d')
    parser.add_argument('--result_path')
    opt = parser.parse_args()
    outputs,labels =  postProcesss(opt.result_path)
    print('Evaluating top_k_accuracy ...')
    top_acc = top_k_accuracy(outputs,labels,topk=(1, 5))
    print(f'\ntop{1}_acc\t{top_acc[0]:.4f}')
    print(f'\ntop{5}_acc\t{top_acc[1]:.4f}')

    print('Evaluating mean_class_accuracy ...')
    mean_acc = mean_class_accuracy(outputs,labels)
    print(f'\nmean_acc\t{mean_acc:.4f}')


# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import os.path as osp
import os
import numpy as np
import torch
import datasets
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from collections import defaultdict
import json


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
            torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def evaluate_all(distmat, query=None, gallery=None,
                    query_ids=None, gallery_ids=None,
                    query_cams=None, gallery_cams=None,
                    cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                    single_gallery_shot=True,
                    first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                    query_cams, gallery_cams, **params)
                    for name, params in cmc_configs.items()}                           
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []
        
    table_dict["value"].extend(
        [{"key": "Number of images", "value": str(15913)},
        {"key": "Number of classes", "value": str(751)}])
    for k in cmc_topk:
        table_dict["value"].append({"key": "Top-" + str(k) + " accuracy",
                                        "value": str('{:.1%}'.format(cmc_scores['market1501'][k - 1]))})
    print('CMC Scores{:>12}'
        .format('market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
            .format(k, cmc_scores['market1501'][k - 1]))
        
    print(table_dict)
    writer = open('PCB_inference_result.json', 'w')
    json.dump(table_dict, writer)
    writer.close()
    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


def load_result(filepath):
    count = 0
    features = OrderedDict()
    for root, dirs, files in os.walk(filepath): 
        for  file in files:
            file_tmp = file.split('.', 2)[0]
            list_file = file_tmp.split('_')
            if list_file[4] == '1': 
                file = filepath + '/' + file
                output = np.fromfile(file, dtype='float32')
                output = torch.from_numpy(output)
                output = output.reshape(2048, 6, 1)
                filename = list_file[0] + '_' + list_file[1] + '_' + list_file[2] + '_' + list_file[3] + '.jpg'
                if list_file[0] == '1488' or filename == '0000_c6s3_094992_01.jpg' \
                or filename == '0000_c4s6_022316_04.jpg' or filename == '0000_c1s6_023071_04.jpg':
                    filename = filename + '.jpg'
                features[filename] = output
                count = count + 1
    return features


def evaluate_Ascend310(query_filepath, gallery_filepath, query, gallery):
    print('extracting query features\n')
    query_features_0 = load_result(query_filepath)
    print('extracting gallery features\n')
    gallery_features_0 = load_result(gallery_filepath)
    distmat = pairwise_distance(query_features_0, gallery_features_0, query, gallery)
    return evaluate_all(distmat, query=query, gallery=gallery)


def main(args):
    dataset = get_data(args.dataset, args.data_dir)
    evaluate_Ascend310(args.query, args.gallery, dataset.query, dataset.gallery)
    return

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")

    parser.add_argument('-q', '--query', type=str, default='./dumpOutput_device0_query')

    parser.add_argument('-g', '--gallery', type=str, default='./dumpOutput_device0_gallery')
    parser.add_argument('-d', '--dataset', type=str, default='market',
                        choices=datasets.names())
                        
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./datasets/Market-1501/')

    main(parser.parse_args())

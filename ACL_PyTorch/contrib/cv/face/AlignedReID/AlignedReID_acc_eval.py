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

import numpy as np
import sys
sys.path.insert(0, './AlignedReID-Re-Production-Pytorch')

from aligned_reid.utils.utils import load_pickle
import time
import os
from sklearn.metrics import average_precision_score
from collections import defaultdict
from contextlib import contextmanager
import sklearn


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def mean_ap(
        distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        average=True):
    """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """

    cur_version = sklearn.__version__
    required_version = '0.18.1'
    if cur_version != required_version:
        print('User Warning: Version {} is required for package scikit-learn, '
              'your current version is {}. '
              'As a result, the mAP score may not be totally correct. '
              'You can try `pip uninstall scikit-learn` '
              'and then `pip install scikit-learn=={}`'.format(
            required_version, cur_version, required_version))
    # -------------------------------------------------------------------------

    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query


def eval_map_cmc(
        q_g_dist,
        q_ids=None, g_ids=None,
        q_cams=None, g_cams=None,
        separate_camera_set=None,
        single_gallery_shot=None,
        first_match_break=None,
        topk=None):
    """Compute CMC and mAP.
    Args:
      q_g_dist: numpy array with shape [num_query, num_gallery], the
        pairwise distance between query and gallery samples
    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query
        samples
      cmc_scores: numpy array with shape [topk], the cmc curve
        averaged across query samples
    """
    # Compute mean AP
    mAP = mean_ap(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = cmc(
        distmat=q_g_dist,
        query_ids=q_ids, gallery_ids=g_ids,
        query_cams=q_cams, gallery_cams=g_cams,
        separate_camera_set=separate_camera_set,
        single_gallery_shot=single_gallery_shot,
        first_match_break=first_match_break,
        topk=topk)
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
          .format(mAP, *cmc_scores[[0, 4, 9]]))
    return mAP, cmc_scores


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(
        distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False,
        average=True):
    """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    assert isinstance(query_ids, np.ndarray)
    assert isinstance(gallery_ids, np.ndarray)
    assert isinstance(query_cams, np.ndarray)
    assert isinstance(gallery_cams, np.ndarray)

    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros([m, topk])
    is_valid_query = np.zeros(m)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
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
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / num_valid_queries
    return ret, is_valid_query


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
      array1: numpy array with shape [m1, n]
      array2: numpy array with shape [m2, n]
      type: one of ['cosine', 'euclidean']
    Returns:
      numpy array with shape [m1, m2]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)

        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """ The following naming, e.g. gallery_num, is different from outer scope.
      Don't care about it.
    """
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        """k-reciprocal neighbors"""
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


@contextmanager
def measure_time(enter_msg):
    st = time.time()
    print(enter_msg)
    yield
    print('Done, {:.2f}s'.format(time.time() - st))


"""Accuracy calculation"""
def main(pth, gt_path, key_idx):
    ids = []
    cams = []

    partition_file = gt_path
    partitions = load_pickle(partition_file)
    im_names = partitions['test_im_names']
    global_feats = np.zeros((len(im_names), 2048))
    for name in im_names:
        id = int(name[:8])
        ids.append(id)
        cam = int(name[9:13])
        cams.append(cam)

    marks = partitions['test_marks']
    key = f'_{key_idx}'

    i = 0

    print("========Image index matching========")
    match_t = time.time()
    for root, dirs, files in os.walk(pth):
        for name in files:
            if key in name:
                f = open(os.path.join(root, name), "r")
                global_feat = f.readline()
                global_feat = global_feat.split()
                global_feat = list(map(float, global_feat))
                for idx, im in enumerate(im_names):
                    if im[:-4] == name[:-6]:
                        i += 1
                        print("\r %d / 31969" % i, end='', flush=True)
                        global_feats[idx] = global_feat
                        break

    print("time:%.2f" % (time.time() - match_t))

    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)

    global_feats = normalize(global_feats, axis=1)
    q_inds = marks == 0
    g_inds = marks == 1

    #A helper function just for avoiding code duplication.
    def compute_score(dist_mat):
        mAP, cmc_scores = eval_map_cmc(
            q_g_dist=dist_mat,
            q_ids=ids[q_inds], g_ids=ids[g_inds],
            q_cams=cams[q_inds], g_cams=cams[g_inds],
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True,
            topk=10)
        return mAP, cmc_scores

    with measure_time('Computing global distance...'):
        # query-gallery distance using global distance
        global_q_g_dist = compute_dist(
            global_feats[q_inds], global_feats[g_inds], type='euclidean')

    with measure_time('Computing scores for Global Distance...'):
        mAP, cmc_scores = compute_score(global_q_g_dist)

    with measure_time('Re-ranking...'):
        # query-query distance using global distance
        global_q_q_dist = compute_dist(
            global_feats[q_inds], global_feats[q_inds], type='euclidean')

        # gallery-gallery distance using global distance
        global_g_g_dist = compute_dist(
            global_feats[g_inds], global_feats[g_inds], type='euclidean')

        # re-ranked global query-gallery distance
        re_r_global_q_g_dist = re_ranking(
            global_q_g_dist, global_q_q_dist, global_g_g_dist)

    with measure_time('Computing scores for re-ranked Global Distance...'):
        mAP, cmc_scores = compute_score(re_r_global_q_g_dist)


if __name__ == "__main__":
    result_path = sys.argv[1]
    gt_path = sys.argv[2]
    key_idx = sys.argv[3]
    main(result_path, gt_path, key_idx)

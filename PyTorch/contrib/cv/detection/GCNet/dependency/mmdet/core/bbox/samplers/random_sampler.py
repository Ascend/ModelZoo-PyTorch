# # Copyright 2021 Huawei Technologies Co., Ltd
# #
# # Licensed under the Apache License, Version 2.0 (the License);
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #

# import torch

# from ..builder import BBOX_SAMPLERS
# from .base_sampler import BaseSampler


# @BBOX_SAMPLERS.register_module()
# class RandomSampler(BaseSampler):
#     """Random sampler.

#     Args:
#         num (int): Number of samples
#         pos_fraction (float): Fraction of positive samples
#         neg_pos_up (int, optional): Upper bound number of negative and
#             positive samples. Defaults to -1.
#         add_gt_as_proposals (bool, optional): Whether to add ground truth
#             boxes as proposals. Defaults to True.
#     """

#     def __init__(self,
#                  num,
#                  pos_fraction,
#                  neg_pos_ub=-1,
#                  add_gt_as_proposals=True,
#                  **kwargs):
#         from mmdet.core.bbox import demodata
#         super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
#                                             add_gt_as_proposals)
#         self.rng = demodata.ensure_rng(kwargs.get('rng', None))

#     def random_choice(self, gallery, num):
#         """Random select some elements from the gallery.

#         If `gallery` is a Tensor, the returned indices will be a Tensor;
#         If `gallery` is a ndarray or list, the returned indices will be a
#         ndarray.

#         Args:
#             gallery (Tensor | ndarray | list): indices pool.
#             num (int): expected sample num.

#         Returns:
#             Tensor or ndarray: sampled indices.
#         """
#         assert len(gallery) >= num

#         is_tensor = isinstance(gallery, torch.Tensor)
#         if not is_tensor:
#             if torch.cuda.is_available():
#                 device = torch.cuda.current_device()
#             else:
#                 device = 'cpu'
#             gallery = torch.tensor(gallery, dtype=torch.long, device=device)
#         perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
#         rand_inds = gallery[perm]
#         if not is_tensor:
#             rand_inds = rand_inds.cpu().numpy()
#         return rand_inds

#     def _sample_pos(self, assign_result, num_expected, **kwargs):
#         """Randomly sample some positive samples."""
#         pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
# #         print('========assign_result:',assign_result.gt_inds.size())
# #         print('========pos_num:',pos_inds.numel())
#         if pos_inds.numel() != 0:
#             pos_inds = pos_inds.squeeze(1)
#         else:
#             return torch.zeros_like(assign_result.gt_inds).int()
#         if pos_inds.numel() <= num_expected:
#             pos_mask = torch.zeros_like(assign_result.gt_inds).int()
#             pos_mask[pos_inds] = 1
#             return pos_mask
#         else:
#             rand_inds = self.random_choice(pos_inds, num_expected)
#             pos_mask = torch.zeros_like(assign_result.gt_inds).int()
#             pos_mask[rand_inds] = 1
#             return pos_mask
# #             return self.random_choice(pos_inds, num_expected)

#     def _sample_neg(self, assign_result, num_expected, **kwargs):
#         """Randomly sample some negative samples."""
#         neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
# #         print('========assign_result:',assign_result.gt_inds.size())
# #         print('=========neg_num:',neg_inds.numel())
#         if neg_inds.numel() != 0:
#             neg_inds = neg_inds.squeeze(1)
#         else:
#             return torch.zeros_like(assign_result.gt_inds).int()
#         if len(neg_inds) <= num_expected:
#             neg_mask = torch.zeros_like(assign_result.gt_inds).int()
#             neg_mask[neg_inds] = 1
#             return neg_mask
#         else:
#             rand_inds = self.random_choice(neg_inds, num_expected)
#             neg_mask = torch.zeros_like(assign_result.gt_inds).int()
# #             print(torch.npu.synchronize(),'==================R5')
#             print(neg_mask.size())
#             print(rand_inds.size())
#             neg_mask[rand_inds] = 1
# #             print(torch.npu.synchronize(),'==================R6')
#             return neg_mask
# #             return self.random_choice(neg_inds, num_expected)

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    # def random_choice(self, gallery, num):
    #     """Random select some elements from the gallery.

    #     If `gallery` is a Tensor, the returned indices will be a Tensor;
    #     If `gallery` is a ndarray or list, the returned indices will be a
    #     ndarray.

    #     Args:
    #         gallery (Tensor | ndarray | list): indices pool.
    #         num (int): expected sample num.

    #     Returns:
    #         Tensor or ndarray: sampled indices.
    #     """
    #     assert len(gallery) >= num

    #     is_tensor = isinstance(gallery, torch.Tensor)
    #     if not is_tensor:
    #         if torch.npu.is_available():
    #             device = torch.cuda.current_device()
    #         else:
    #             device = 'cpu'
    #         gallery = torch.tensor(gallery, dtype=torch.long, device=device)
    #     perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
    #     rand_inds = gallery[perm]
    #     if not is_tensor:
    #         rand_inds = rand_inds.cpu().numpy()
    #     return rand_inds

    def random_choice(self,gallery, num):
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.npu.is_available():
                device = torch.npu.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    # def _sample_pos(self, assign_result, num_expected, **kwargs):
    #     """Randomly sample some positive samples."""
    #     #<AssignResult(num_gts=40, gt_inds.shape=(268569,), max_overlaps.shape=(268569,), labels=None)>
    #     #num_expected:128
    #     pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)#pos_inds the way come from
    #     pos_inds=pos_inds.long()
    #     if pos_inds.numel() != 0:
    #         pos_inds = pos_inds.squeeze(1)
    #     if pos_inds.numel() <= num_expected:
    #         return pos_inds
    #     else:
    #         return self.random_choice(pos_inds, num_expected)

    def _sample_pos(self,assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        assign_result_gt_inds =assign_result.gt_inds.int()
        gt_inds_zero = torch.zeros_like(assign_result_gt_inds)
        pos_inds = torch.nonzero(assign_result_gt_inds > 0, as_tuple=False)
#         print('num exp:',num_expected)
#         if num_expected == 64:
#             print('pos sample num: ',pos_inds.size())
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            pass
        else:
            pos_inds =self.random_choice(pos_inds, num_expected)
    
        if not  min(pos_inds.shape) == 0:
            gt_inds_zero[pos_inds] = 1
        return gt_inds_zero

    # def _sample_neg(self, assign_result, num_expected, **kwargs):
    #     """Randomly sample some negative samples."""
    #     neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
    #     neg_inds=neg_inds.long()
    #     if neg_inds.numel() != 0:
    #         neg_inds = neg_inds.squeeze(1)
    #     if len(neg_inds) <= num_expected:
    #         return neg_inds
    #     else:
    #         return self.random_choice(neg_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        assign_result_gt_inds = assign_result.gt_inds.int()
        gt_inds_zero = torch.zeros_like(assign_result_gt_inds)
        neg_inds = torch.nonzero(assign_result_gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if neg_inds.numel() <= num_expected:
            pass
        else:
            neg_inds =self.random_choice(neg_inds, num_expected)
        if not min(neg_inds.shape) == 0:
            gt_inds_zero[neg_inds] = 1
        return gt_inds_zero

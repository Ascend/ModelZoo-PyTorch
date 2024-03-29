# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

from mmocr.utils import is_2dlist, is_type_list


@DATASETS.register_module()
class UniformConcatDataset(ConcatDataset):
    """A wrapper of ConcatDataset which support dataset pipeline assignment and
    replacement.

    Args:
        datasets (list[dict] | list[list[dict]]): A list of datasets cfgs.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
        show_mean_scores (str | bool): Whether to compute the mean evaluation
            results, only applicable when ``separate_eval=True``. Options are
            [True, False, ``auto``]. If ``True``, mean results will be added to
            the result dictionary with keys in the form of
            ``mean_{metric_name}``. If 'auto', mean results will be shown only
            when more than 1 dataset is wrapped.
        pipeline (None | list[dict] | list[list[dict]]): If ``None``,
            each dataset in datasets use its own pipeline;
            If ``list[dict]``, it will be assigned to the dataset whose
            pipeline is None in datasets;
            If ``list[list[dict]]``, pipeline of dataset which is None
            in datasets will be replaced by the corresponding pipeline
            in the list.
        force_apply (bool): If True, apply pipeline above to each dataset
            even if it have its own pipeline. Default: False.
    """

    def __init__(self,
                 datasets,
                 separate_eval=True,
                 show_mean_scores='auto',
                 pipeline=None,
                 force_apply=False,
                 **kwargs):
        new_datasets = []
        if pipeline is not None:
            assert isinstance(
                pipeline,
                list), 'pipeline must be list[dict] or list[list[dict]].'
            if is_type_list(pipeline, dict):
                self._apply_pipeline(datasets, pipeline, force_apply)
                new_datasets = datasets
            elif is_2dlist(pipeline):
                assert is_2dlist(datasets)
                assert len(datasets) == len(pipeline)
                for sub_datasets, tmp_pipeline in zip(datasets, pipeline):
                    self._apply_pipeline(sub_datasets, tmp_pipeline,
                                         force_apply)
                    new_datasets.extend(sub_datasets)
        else:
            if is_2dlist(datasets):
                for sub_datasets in datasets:
                    new_datasets.extend(sub_datasets)
            else:
                new_datasets = datasets
        datasets = [build_dataset(c, kwargs) for c in new_datasets]
        super().__init__(datasets, separate_eval)

        if not separate_eval:
            raise NotImplementedError(
                'Evaluating datasets as a whole is not'
                ' supported yet. Please use "separate_eval=True"')

        assert isinstance(show_mean_scores, bool) or show_mean_scores == 'auto'
        if show_mean_scores == 'auto':
            show_mean_scores = len(self.datasets) > 1
        self.show_mean_scores = show_mean_scores
        if show_mean_scores is True or show_mean_scores == 'auto' and len(
                self.datasets) > 1:
            if len(set([type(ds) for ds in self.datasets])) != 1:
                raise NotImplementedError(
                    'To compute mean evaluation scores, all datasets'
                    'must have the same type')

    @staticmethod
    def _apply_pipeline(datasets, pipeline, force_apply=False):
        from_cfg = all(isinstance(x, dict) for x in datasets)
        assert from_cfg, 'datasets should be config dicts'
        assert all(isinstance(x, dict) for x in pipeline)
        for dataset in datasets:
            if dataset['pipeline'] is None or force_apply:
                dataset['pipeline'] = copy.deepcopy(pipeline)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: Results of each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1

            total_eval_results = dict()

            if self.show_mean_scores:
                mean_eval_results = defaultdict(list)

            for dataset in self.datasets:
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluating {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})
                    if self.show_mean_scores:
                        mean_eval_results[k].append(v)

            if self.show_mean_scores:
                for k, v in mean_eval_results.items():
                    total_eval_results[f'mean_{k}'] = np.mean(v)

            return total_eval_results
        else:
            raise NotImplementedError(
                'Evaluating datasets as a whole is not'
                ' supported yet. Please use "separate_eval=True"')

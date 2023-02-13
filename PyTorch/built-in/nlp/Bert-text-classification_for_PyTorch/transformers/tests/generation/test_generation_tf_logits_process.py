# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np

from transformers import is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf

    from transformers.generation_tf_logits_process import (
        TFLogitsProcessorList,
        TFMinLengthLogitsProcessor,
        TFNoBadWordsLogitsProcessor,
        TFNoRepeatNGramLogitsProcessor,
        TFRepetitionPenaltyLogitsProcessor,
        TFTemperatureLogitsWarper,
        TFTopKLogitsWarper,
        TFTopPLogitsWarper,
    )
    from transformers.tf_utils import set_tensor_by_indices_to_value

    from ..test_modeling_tf_common import ids_tensor


@require_tf
class TFLogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = np.ones((batch_size, length), dtype=np.float32) / length
        return scores

    def test_min_length_dist_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        min_dist_processor = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)

        # check that min length is applied at length 5
        input_ids = ids_tensor((batch_size, 5), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertListEqual(scores_before_min_length[:, eos_token_id].numpy().tolist(), 4 * [-float("inf")])

        # check that min length is not applied anymore at length 15
        input_ids = ids_tensor((batch_size, 15), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(scores_before_min_length)).numpy())

    def test_temperature_dist_warper(self):
        input_ids = None
        length = 20

        scores = self._get_uniform_logits(batch_size=2, length=length)

        # tweak scores to not be uniform anymore
        scores[1, 5] = (1 / length) + 0.1  # peak, 1st batch
        scores[1, 10] = (1 / length) - 0.4  # valley, 1st batch

        # compute softmax
        probs = tf.nn.softmax(scores, axis=-1)

        temp_dist_warper_sharper = TFTemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = TFTemperatureLogitsWarper(temperature=1.3)

        warped_prob_sharp = tf.nn.softmax(temp_dist_warper_sharper(input_ids, tf.identity(scores)), axis=-1)
        warped_prob_smooth = tf.nn.softmax(temp_dist_warper_smoother(input_ids, tf.identity(scores)), axis=-1)

        # uniform distribution stays uniform
        tf.debugging.assert_near(probs[0, :], warped_prob_sharp[0, :], atol=1e-3)
        tf.debugging.assert_near(probs[0, :], warped_prob_smooth[0, :], atol=1e-3)

        # sharp peaks get higher, valleys get lower
        self.assertLess(tf.math.reduce_max(probs[1, :]), tf.math.reduce_max(warped_prob_sharp[1, :]))
        self.assertGreater(tf.math.reduce_min(probs[1, :]), tf.math.reduce_min(warped_prob_sharp[1, :]))

        # smooth peaks get lower, valleys get higher
        self.assertGreater(tf.math.reduce_max(probs[1, :]), tf.math.reduce_max(warped_prob_smooth[1, :]))
        self.assertLess(tf.math.reduce_min(probs[1, :]), tf.math.reduce_min(warped_prob_smooth[1, :]))

    def test_repetition_penalty_dist_process(self):
        input_ids = tf.constant([[0, 1], [5, 0]], dtype=tf.int32)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        mask = tf.cast(tf.constant([[1] + 9 * [0], 10 * [0]]), tf.bool)
        scores = set_tensor_by_indices_to_value(scores, mask, -1 / vocab_size)
        mask = tf.cast(tf.constant([10 * [0], 5 * [0] + [1] + 4 * [0]]), tf.bool)
        scores = set_tensor_by_indices_to_value(scores, mask, 4 / vocab_size)

        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)

        scores = rep_penalty_proc(input_ids, tf.identity(scores))

        # check that values were correctly changed
        self.assertAlmostEqual(scores[0, 0].numpy(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[0, 1].numpy(), (1 / vocab_size) / 2)

        self.assertAlmostEqual(scores[1, 0].numpy(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[1, 5].numpy(), (4 / vocab_size) / 2)

    def test_top_k_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = np.broadcast_to(np.arange(vocab_size)[None, :], (batch_size, vocab_size)).copy()
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size

        top_k_warp = TFTopKLogitsWarper(3)

        scores = top_k_warp(input_ids, ramp_logits)

        # check that correct tokens are filtered
        self.assertListEqual(tf.math.is_inf(scores[0]).numpy().tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual(tf.math.is_inf(scores[1]).numpy().tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        top_k_warp_safety_check = TFTopKLogitsWarper(top_k=1, filter_value=0.0, min_tokens_to_keep=3)

        scores = top_k_warp_safety_check(input_ids, logits)
        # uniform dist is not changed
        self.assertListEqual(tf.math.reduce_sum(tf.where(scores == 0.0, 1, 0), axis=-1).numpy().tolist(), [0, 0])

        ramp_logits = np.broadcast_to(np.arange(length)[None, :], (batch_size, length)).copy()
        scores = top_k_warp_safety_check(input_ids, ramp_logits)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual(tf.math.reduce_sum(tf.where(scores == 0.0, 1, 0), axis=-1).numpy().tolist(), [2, 2])

    def test_top_p_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TFTopPLogitsWarper)
        dist = np.log(np.array([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]], dtype=np.float32))

        top_p_warp = TFTopPLogitsWarper(0.7)
        filtered_dist = tf.exp(top_p_warp(input_ids, dist))

        # dist should be filtered to keep min num values so that sum is >= 0.7
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = tf.constant([[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]], dtype=tf.float32)
        tf.debugging.assert_near(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3)

        # check edge cases with negative and extreme logits
        ramp_logits = np.broadcast_to(
            np.arange(vocab_size, dtype=np.float32)[None, :], (batch_size, vocab_size)
        ).copy() - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        top_p_warp = TFTopPLogitsWarper(0.9, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = top_p_warp(input_ids, ramp_logits)

        # first batch should keep three tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps
        # 2.
        self.assertListEqual(
            tf.math.reduce_sum(tf.where(filtered_dist != 0.0, 1, 0), axis=-1).numpy().tolist(), [3, 2]
        )

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2

        input_ids = tf.constant([[1, 1, 2, 1], [0, 1, 0, 1]], dtype=tf.int32)
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = TFNoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = TFNoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, tf.identity(scores))
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, tf.identity(scores))

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_2_gram).numpy().tolist(), [[False, True, True], [True, False, False]]
        )

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_3_gram).numpy().tolist(), [[False, False, False], [True, False, False]]
        )

    def test_no_bad_words_dist_processor(self):
        vocab_size = 5
        batch_size = 2
        eos_token_id = 4

        input_ids = tf.constant([[0, 1, 3, 1], [0, 1, 0, 1]], dtype=tf.int32)
        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=bad_word_tokens, eos_token_id=eos_token_id)

        filtered_scores = no_bad_words_dist_proc(input_ids, tf.identity(scores))

        # batch 1: 1st, 2nd, and 4th (0, 1, 3) token are forbidden
        # batch 2: 1st, 2nd, and 3rd (0, 1, 2) token are forbidden
        self.assertListEqual(
            tf.math.is_inf(filtered_scores).numpy().tolist(),
            [[True, True, False, True, True], [True, True, True, False, True]],
        )

    def test_processor_list(self):
        batch_size = 4
        sequence_length = 10
        vocab_size = 15
        eos_token_id = 0

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, sequence_length), vocab_size)
        input_ids_comp = tf.identity(input_ids)

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = tf.identity(scores)

        # instantiate all dist processors
        min_dist_proc = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        temp_dist_warp = TFTemperatureLogitsWarper(temperature=0.5)
        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)
        top_k_warp = TFTopKLogitsWarper(3)
        top_p_warp = TFTopPLogitsWarper(0.8)
        no_repeat_proc = TFNoRepeatNGramLogitsProcessor(2)
        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=[[1]], eos_token_id=eos_token_id)

        # no processor list
        scores = min_dist_proc(input_ids, scores)
        scores = temp_dist_warp(input_ids, scores)
        scores = rep_penalty_proc(input_ids, scores)
        scores = top_k_warp(input_ids, scores)
        scores = top_p_warp(input_ids, scores)
        scores = no_repeat_proc(input_ids, scores)
        scores = no_bad_words_dist_proc(input_ids, scores)

        # with processor list
        processor = TFLogitsProcessorList(
            [
                min_dist_proc,
                temp_dist_warp,
                rep_penalty_proc,
                top_k_warp,
                top_p_warp,
                no_repeat_proc,
                no_bad_words_dist_proc,
            ]
        )
        scores_comp = processor(input_ids, scores_comp)

        # remove inf
        scores = set_tensor_by_indices_to_value(scores, tf.math.is_inf(scores), -1e9)
        scores_comp = set_tensor_by_indices_to_value(scores_comp, tf.math.is_inf(scores_comp), -1e9)

        # scores should be equal
        tf.debugging.assert_near(scores, scores_comp, atol=1e-3)

        # input_ids should never be changed
        self.assertListEqual(input_ids.numpy().tolist(), input_ids_comp.numpy().tolist())

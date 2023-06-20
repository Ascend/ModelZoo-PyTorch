# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Parallel beam search module."""

import logging
from typing import (
    Any,
    Dict,
    List,
    Tuple
)
import numpy as np

from espnet_onnx.utils.function import (
    topk,
    end_detect
)
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.beam_search.hyps import (
    Hypothesis,
    BatchHypothesis
)


class MultiBatchBeamSearch(BatchBeamSearch):
    """Multi Batch beam search implementation."""

    def merge_batch(self, hyps_list: List[BatchHypothesis]) -> BatchHypothesis:
        if len(hyps_list) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=np.concatenate([h.yseq for h in hyps_list]),
            length=np.concatenate([h.length for h in hyps_list]),
            score=np.concatenate([h.score for h in hyps_list]),
            scores={k: np.concatenate([h.scores[k] for h in hyps_list])
                    for k in self.scorers},
            states={k: np.concatenate([h.states[k] for h in hyps_list]) for k in self.scorers},
        )

    def merge_partial_states(self, states, max_len):
        out_states = []
        out_states.append(
            np.concatenate([np.pad(state[0], ((0, max_len-state[0].shape[0]), (0,0), (0,0), (0,0)))
                            if state[0].shape[0] < max_len else state[0] for state in states], axis=2)
        )
        out_states.append(
            np.concatenate([state[1] for state in states], axis=0)
        )
        out_states.append([state[2] for state in states])
        out_states.append([state[3] for state in states])
        out_states.append(
            np.concatenate([state[4] for state in states], axis=0)
        )
        return tuple(out_states)

    def search(self, running_hyps: BatchHypothesis, x: np.ndarray, remined_list = None) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (np.ndarray): Encoded speech feature (B, T, D)
            remined_list (list): list for remined batch idx
        Returns:
            BatchHypothesis: Best sorted hypotheses
        """
        init_batch = x.shape[0]
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = np.zeros(
            (n_batch, self.n_vocab), dtype=x.dtype
        )
        scores, states = self.score_full(running_hyps, np.vstack(
            [np.vstack([x_i for _ in range(n_batch // init_batch)]) for x_i in x]).reshape(n_batch, *x.shape[1:]))
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = topk(pre_beam_scores, self.pre_beam_size)
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x, remined_list=remined_list)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += np.expand_dims(running_hyps.score, 1)

        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids, init_batch)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id]
                            for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(
                                v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)

    def multi_post_process(
            self,
            i: int,
            init_batch: int,
            maxlen: int,
            running_hyps: BatchHypothesis,
            ended_hyps_list: List[List[Hypothesis]],
            keep_list: List[int],
            dynamic_search: bool
    ):
        sample_num = len(running_hyps) // init_batch
        results = []
        hyps_list = []

        for batch_idx in range(init_batch):
            idxes = list(range(batch_idx*sample_num, (batch_idx+1)*sample_num))
            _hyps = self._batch_select(running_hyps, idxes)
            end_idx = batch_idx
            if dynamic_search:
                end_idx = keep_list[batch_idx]
            if not dynamic_search and batch_idx not in keep_list:
                hyps_list.append(_hyps)
            else:
                _hyps = self._batch_select(running_hyps, idxes)
                remined_running_hyps, all_running_hyps = self.post_process(
                    i, maxlen, _hyps, ended_hyps_list[end_idx], keep_ori_hyps=True)
                results.append(
                    (idxes, remined_running_hyps)
                )
                hyps_list.append(all_running_hyps)
        if not dynamic_search and i == maxlen - 1:
            n_batch = hyps_list[0].yseq.shape[0]
            for _idx, _hyps in enumerate(hyps_list):
                if _idx not in keep_list:
                    yseq_eos = np.hstack(
                        (
                            _hyps.yseq,
                            np.full(
                                (n_batch, 1),
                                self.eos,
                                dtype=np.int64,
                            )
                        )
                    )
                    _hyps.yseq.resize(yseq_eos.shape, refcheck=False)
                    _hyps.yseq[:] = yseq_eos
                    _hyps.length[:] = yseq_eos.shape[1]

        hyps_list = self.merge_batch(hyps_list)
        return results, hyps_list

    def batch_beam(
            self, weighted_scores: np.ndarray, ids: np.ndarray, init_batch: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch-compute topk full token ids and partial token ids.
        Args:
            weighted_scores (np.ndarray): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (np.ndarray): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.
            init_batch (int): batch for encoded speech feature
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`
        """
        length = weighted_scores.shape[0] // init_batch
        offsets = np.expand_dims(np.arange(init_batch), axis=-1) * self.n_vocab * length
        top_ids = topk(weighted_scores.reshape(init_batch, -1), self.beam_size)
        top_ids = (top_ids + offsets).reshape(-1)
        prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: np.ndarray) -> BatchHypothesis:
        """Get an initial hypothesis data.
        Args:
            x (np.ndarray): The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        batch_num = x.shape[0]
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            if k in ['ctc']:
                init_states[k] = d.multi_batch_init_state(x)
            else:
                init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
        return self.batchfy([
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=np.array([self.sos], dtype=np.int64),
            )] * batch_num)

    def __call__(
        self, x: np.ndarray,
        dynamic_search: bool = True
    ) -> List[Hypothesis]:
        """Perform beam search.
        Args:
            x (np.ndarray): Encoded speech feature (B, T, D)
        Returns:
            list[Hypothesis]: N-best decoding results
        """
        # set length bounds
        if self.maxlenratio == 0:
            maxlen = x.shape[1]
        elif self.maxlenratio < 0:
            maxlen = -1 * int(self.maxlenratio)
        else:
            maxlen = max(1, int(self.maxlenratio * x.shape[1]))
        minlen = int(self.minlenratio * x.shape[0])
        logging.debug("decoder input length: " + str(x.shape[1]))
        logging.debug("max output length: " + str(maxlen))
        logging.debug("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        init_batch = x.shape[0]
        ended_hyps_list = [[] for _ in range(init_batch)]
        remined_list = list(range(init_batch))

        init_batch = x.shape[0]
        for i in range(maxlen):
            logging.debug("position " + str(i))
            if not dynamic_search:
                best = self.search(running_hyps, x)
            else:
                best = self.search(running_hyps, x[remined_list], remined_list)
                init_batch = len(remined_list)
            # post process of one iteration
            remined_running_hyps_list, running_hyps = self.multi_post_process(
                i, init_batch, maxlen, best, ended_hyps_list, remined_list, dynamic_search)
            # end detection
            remined_idx = 0
            remined_idxes = []
            for _idx in remined_list.copy():
                running_idxes, remined_running_hyps = remined_running_hyps_list[remined_idx]
                ended_hyps = ended_hyps_list[_idx]

                if self.maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                    logging.debug(f"End detected at {i}")
                    remined_list.remove(_idx)
                elif len(remined_running_hyps) == 0:
                    logging.debug("No hypothesis. Finish decoding.")
                    remined_list.remove(_idx)
                else:
                    remined_idxes += running_idxes
                    logging.debug(f"Remained hypotheses: {len(running_hyps)}")
                remined_idx += 1
            if not remined_list:
                break
            if dynamic_search:
                running_hyps = self._batch_select(running_hyps, remined_idxes)

        nbest_hyps_list = []
        for ended_hyps in ended_hyps_list:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
            # check the number of hypotheses reaching to eos
            if len(nbest_hyps) == 0:
                logging.warning(
                    "There is no N-best results, perform recognition "
                    "again with smaller minlenratio."
                )
                return (
                    []
                    if self.minlenratio < 0.1
                    else self(x[_idx], self.maxlenratio, max(0.0, self.minlenratio - 0.1))
                )

            # report the best result
            best = nbest_hyps[0]
            for k, v in best.scores.items():
                logging.debug(
                    f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
                )
            logging.debug(f"Total log probability: {best.score:.2f}")
            logging.debug(
                f"Normalized log probability: {best.score / len(best.yseq):.2f}")
            logging.debug(f"Total number of ended hypotheses: {len(nbest_hyps)}")
            if self.token_list is not None:
                logging.debug(
                    "Best hypo: "
                    + "".join([self.token_list[int(x)] for x in best.yseq[1:-1]])
                    + "\n"
                )
            nbest_hyps_list.append(nbest_hyps)
        return nbest_hyps_list

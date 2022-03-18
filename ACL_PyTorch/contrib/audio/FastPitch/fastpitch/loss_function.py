# Copyright 2020 Huawei Technologies Co., Ltd
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

import torch
import torch.nn.functional as F
from torch import nn

from common.utils import mask_from_lens
from fastpitch.attn_loss_function import AttentionCTCLoss


class FastPitchLoss(nn.Module):
    def __init__(self, dur_predictor_loss_scale=1.0,
                 pitch_predictor_loss_scale=1.0, attn_loss_scale=1.0,
                 energy_predictor_loss_scale=0.1):
        super(FastPitchLoss, self).__init__()
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale
        self.energy_predictor_loss_scale = energy_predictor_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, pitch_tgt,
         energy_pred, energy_tgt, attn_soft, attn_hard, attn_dur,
         attn_logprob) = model_out

        (mel_tgt, in_lens, out_lens) = targets

        dur_tgt = attn_dur
        dur_lens = in_lens

        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        dur_mask_sum = dur_mask.sum()

        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask_sum

        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)

        mel_mask = mel_tgt.ne(0).float()
        mel_mask_sum = mel_mask.sum()

        loss_fn = F.mse_loss
        mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask_sum

        ldiff = pitch_tgt.size(2) - pitch_pred.size(2)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask.unsqueeze(1)).sum() / dur_mask_sum

        if energy_pred is not None:
            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction='none')
            energy_loss = (energy_loss * dur_mask).sum() / dur_mask_sum
        else:
            energy_loss = 0

        # Attention loss
        attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        loss = (mel_loss
                + dur_pred_loss * self.dur_predictor_loss_scale
                + pitch_loss * self.pitch_predictor_loss_scale
                + energy_loss * self.energy_predictor_loss_scale
                + attn_loss * self.attn_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'energy_loss': energy_loss.clone().detach(),
            'attn_loss': attn_loss.clone().detach(),
            'dur_mask_sum': dur_mask_sum.clone().detach(),
            'mel_mask_sum': mel_mask_sum.clone().detach(),
            'dur_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask_sum).detach(),
        }

        if energy_pred is not None:
            meta['energy_loss'] = energy_loss.clone().detach()

        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k, v in meta.items()}
        return loss, meta

#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#
# Copyright (c) Runpei Dong, ArChip Lab.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss

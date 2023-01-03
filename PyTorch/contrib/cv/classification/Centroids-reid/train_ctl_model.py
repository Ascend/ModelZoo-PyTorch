
# encoding: utf-8
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
"""
Adapted and extended by:
@author: mikwieczorek
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
if torch.__version__ >= "1.8":
     import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities import AttributeDict, rank_zero_only
from torch import tensor
from tqdm import tqdm

from config import cfg
from modelling.bases import ModelBase
from utils.misc import run_main
import time

class CTLModel(ModelBase):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.losses_names = [
            "query_xent",
            "query_triplet",
            "query_center",
            "centroid_triplet",
        ]
        self.losses_dict = {n: [] for n in self.losses_names}

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        torch.npu.synchronize()
        start = time.time()

        opt, opt_center = self.optimizers(use_pl_optimizer=True)

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if self.trainer.current_epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0,
                    float(self.trainer.current_epoch + 1)
                    / float(self.hparams.SOLVER.WARMUP_EPOCHS),
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR

        opt_center.zero_grad()
        opt.zero_grad()

        x, class_labels, camid, isReal = batch  # batch is a tuple

        unique_classes = len(np.unique(class_labels.detach().cpu()))

        # Get backbone features
        _, features = self.backbone(x)

        # query
        contrastive_loss_query, _, _ = self.contrastive_loss(
            features, class_labels, mask=isReal
        )
        contrastive_loss_query = (
            contrastive_loss_query * self.hparams.SOLVER.QUERY_CONTRASTIVE_WEIGHT
        )

        class_labels_real = class_labels[isReal]
        features_real = features[isReal]
        center_loss = self.hparams.SOLVER.CENTER_LOSS_WEIGHT * self.center_loss(
            features_real, class_labels_real
        )
        bn_features = self.bn(features_real)
        cls_score = self.fc_query(bn_features)
        xent_query = self.xent(cls_score, class_labels_real)
        xent_query = xent_query * self.hparams.SOLVER.QUERY_XENT_WEIGHT

        # Prepare masks for uneven numbe of sample per pid in a batch
        ir = isReal.view(unique_classes, -1)
        t = repeat(ir, "c b -> c b s", s=self.hparams.DATALOADER.NUM_INSTANCE)
        t_re = rearrange(t, "c b s -> b (c s)")
        t_re = t_re & isReal

        masks, labels_list = self.create_masks_train(class_labels)  ## True for gallery
        masks = masks.to(features.device)
        t_re = t_re.npu()
        masks = masks & t_re

        masks_float = masks.float().to(features.device)
        padded = masks_float.unsqueeze(-1) * features.unsqueeze(0)  # For broadcasting

        centroids_mask = rearrange(
            masks, "i (ins s) -> i ins s", s=self.hparams.DATALOADER.NUM_INSTANCE
        )
        padded_tmp = rearrange(
            padded,
            "i (ins s) dim -> i ins s dim",
            s=self.hparams.DATALOADER.NUM_INSTANCE,
        )
        valid_inst = centroids_mask.sum(-1)
        valid_inst_bool = centroids_mask.sum(-1).bool()
        centroids_emb = padded_tmp.sum(-2) / valid_inst.masked_fill(
            valid_inst == 0, 1
        ).unsqueeze(-1)

        contrastive_loss_total = []
        ap_total = []
        an_total = []
        l2_mean_norm_total = []
        xent_centroids_total = []

        for i in range(self.hparams.DATALOADER.NUM_INSTANCE):
            if valid_inst_bool[i].sum() <= 1:
                continue

            current_mask = masks[i, :]
            current_labels = class_labels[~current_mask & t_re[i]]
            query_feat = features[~current_mask & t_re[i]]
            current_centroids = centroids_emb[i]
            current_centroids = current_centroids[
                torch.abs(current_centroids).sum(1) > 1e-7
            ]
            embeddings_concat = torch.cat((query_feat, current_centroids))
            labels_concat = torch.cat((current_labels, current_labels))

            contrastive_loss, dist_ap, dist_an = self.contrastive_loss(
                embeddings_concat, labels_concat
            )

            with torch.no_grad():
                dist_ap = dist_ap.data.mean()
                dist_an = dist_an.data.mean()
            ap_total.append(dist_ap)
            an_total.append(dist_an)

            contrastive_loss_total.append(contrastive_loss)

            # L2 norm of centroid vectors
            l2_mean_norm = torch.norm(current_centroids, dim=1).mean()
            l2_mean_norm_total.append(l2_mean_norm)

        contrastive_loss_step = (
            torch.mean(torch.stack(contrastive_loss_total))
            * self.hparams.SOLVER.CENTROID_CONTRASTIVE_WEIGHT
        )
        dist_ap = torch.mean(torch.stack(ap_total))
        dist_an = torch.mean(torch.stack(an_total))
        l2_mean_norm_total = torch.mean(torch.stack(l2_mean_norm_total))

        total_loss = (
            contrastive_loss_step + center_loss + xent_query + contrastive_loss_query
        )

        self.manual_backward(total_loss, optimizer=opt)
        opt.step()

        for param in self.center_loss.parameters():
            param.grad.data *= 1.0 / self.hparams.SOLVER.CENTER_LOSS_WEIGHT
        opt_center.step()

        torch.npu.synchronize()
        step_time = time.time() - start

        print('step time: %.3f, total_loss: %.3f, contrastive loss: %.3f, center loss: %.3f, xent_query: %.3f, contrastive loss query: %.3f' % (
                step_time, total_loss.item(), contrastive_loss_step.item(),
                center_loss.item(), xent_query.item(), contrastive_loss_query.item()))

        losses = [
            xent_query,
            contrastive_loss_query,
            center_loss,
            contrastive_loss_step,
        ]
        losses = [item.detach() for item in losses]
        losses = list(map(float, losses))

        for name, loss_val in zip(self.losses_names, losses):
            self.losses_dict[name].append(loss_val)

        log_data = {
            "step_dist_ap": float(dist_ap),
            "step_dist_an": float(dist_an),
            "l2_mean_centroid": float(l2_mean_norm_total),
        }

        return {"loss": total_loss, "other": log_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLT Model Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    logger_save_dir = f"{Path(__file__).stem}"

    run_main(cfg, CTLModel, logger_save_dir)


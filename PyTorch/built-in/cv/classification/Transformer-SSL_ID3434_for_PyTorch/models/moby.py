# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffdist import functional

from apex.contrib.combine_tensors import combine_npu

from apex import amp


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class NpuLinear(nn.Linear):
    def forward(self, input):
        return torch_npu.npu_linear(input, self.weight, self.bias)


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class MoBY(nn.Module):
    def __init__(self,
                 cfg,
                 encoder,
                 encoder_k,
                 contrast_momentum=0.99,
                 contrast_temperature=0.2,
                 contrast_num_negative=4096,
                 proj_num_layers=2,
                 pred_num_layers=2,
                 **kwargs):
        super().__init__()
        
        self.cfg = cfg
        
        self.encoder = encoder
        self.encoder_k = encoder_k
        
        self.contrast_momentum = contrast_momentum
        self.contrast_temperature = contrast_temperature
        self.contrast_num_negative = contrast_num_negative
        
        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = MoBYMLP(in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.projector_k = MoBYMLP(in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.predictor = MoBYMLP(num_layers=pred_num_layers)
        self.predictor_k = MoBYMLP(num_layers=pred_num_layers) # ensure consistant archi of q and k to use fused momentum update

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.predictor.parameters(), self.predictor_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if self.cfg.MODEL.SWIN.NORM_BEFORE_MLP == 'bn':
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.K = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.EPOCHS
        self.k = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.START_EPOCH

        # create the queue
        self.register_buffer("queue1", torch.randn(256, self.contrast_num_negative))
        self.register_buffer("queue2", torch.randn(256, self.contrast_num_negative))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.queue_ptr = 0

        # for fused momentum update
        self.is_fused = False
        self.step = 0

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def _get_fused_params(self, params):
        pg = []
        for v in params:
            if v.data.dtype.is_floating_point:
                pg.append(v.data)
        return combine_npu(pg)

    def _get_fused_params_pg(self, model):
        skip = {}
        skip_keywords = {}
        """
        'relative_position_bias_table' in different group of 'MOBY' and 'SwinTransformer'
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
        """
        has_decay, no_decay = [], []
        for name, param in model.named_parameters():
            if len(param.shape) == 1 or name.endswith('.bias') or (name in skip) or \
                check_keywords_in_name(name, skip_keywords):
                no_decay.append(param)
            else:
                has_decay.append(param)
        return has_decay, no_decay

    @torch.no_grad()
    def _fused_momentum_update_key_encoder(self, optimizer):
        """
        Update the key encoder with fused parameters in optimizer to accelerate on NPU
        """
        _contrast_momentum = 1. - (1. - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        d = _contrast_momentum
        d_inv = 1. - d

        if not self.is_fused:
            self.param_q_fused = optimizer.get_model_combined_params()[0]

            h0, n0 = self._get_fused_params_pg(self.encoder_k)
            h1, n1 = self._get_fused_params_pg(self.projector_k)
            h2, n2 = self._get_fused_params_pg(self.predictor_k)
            self.param_k_fused = combine_npu(h0 + h1 + h2 + n0 + n1 + n2)

            self.is_fused = True

        self.param_k_fused *= d
        self.param_k_fused += self.param_q_fused * d_inv

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.contrast_num_negative % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.contrast_num_negative  # move pointer

        self.queue_ptr = ptr

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q.half(), k.half()]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q.half(), queue.clone().detach().half()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).npu()

        return F.cross_entropy(logits, labels)

    def forward(self, im_1, im_2, optimizer):
        feat_1 = self.encoder(im_1).float()  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder(im_2).float()
        proj_2 = self.projector(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if self.step == 0:
                self._momentum_update_key_encoder()  # update the key encoder
                self.step += 1
            else:
                self._fused_momentum_update_key_encoder(optimizer)

            feat_1_ng = self.encoder_k(im_1).float()  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_2).float()
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # compute loss
        loss = self.contrastive_loss(pred_1, proj_2_ng, self.queue2) \
            + self.contrastive_loss(pred_2, proj_1_ng, self.queue1)

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

        return loss
    
    
class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()
        
        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(NpuLinear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = NpuLinear(in_dim if num_layers == 1 else inner_dim, out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x

# Copyright 2022 Huawei Technologies Co., Ltd
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

import collections
from typing import Callable

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import apex_C
from torch import distributed
from torch.nn.functional import linear, normalize
import torch.nn.functional as functional

from utils.utils_npu import npu_clamp


class PartialFC(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).

    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.

    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).

    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels, optimizer)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 1 
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_mom: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_mom: torch.Tensor
        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_mom",
                tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_mom",
                tensor=torch.empty(0, 0))
            self.register_buffer("weight_index",
                tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    @torch.no_grad()
    def sample(self, 
        labels: torch.Tensor, 
        index_positive: torch.Tensor, 
        optimizer: torch.optim.Optimizer):
        """
        This functions will change the value of labels

        Parameters:
        -----------
        labels: torch.Tensor
            pass
        index_positive: torch.Tensor
            pass
        optimizer: torch.optim.Optimizer
            pass
        """
        positive = torch.unique(labels[index_positive], sorted=True).npu()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).npu()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].npu()
            index = index.sort()[0].npu()
        else:
            index = positive
        self.weight_index = index

        labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        
        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_mom = self.weight_mom[self.weight_index]

        if isinstance(optimizer, torch.optim.SGD):
            # TODO the params of partial fc must be last in the params list
            optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
            optimizer.param_groups[-1]["params"][0] = self.weight_activated
            if hasattr(optimizer._amp_stash, "all_fp32_params"):
                optimizer._amp_stash.all_fp32_params[-1] = self.weight_activated
            optimizer.state[self.weight_activated][
                "momentum_buffer"
            ] = self.weight_activated_mom
        else:
            raise

    @torch.no_grad()
    def update(self):
        """ partial weight to global
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_mom[self.weight_index] = self.weight_activated_mom


    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).

        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()
        self.update()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
            self.last_batch_size, batch_size))

        device = local_embeddings.device
        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size), device=device)
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size, dtype=torch.long, device=device) for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive, optimizer)

        norm_embeddings = normalize(embeddings)
        norm_weight_activated = normalize(self.weight_activated)
        logits = torch_npu.npu_format_cast(functional.linear(norm_embeddings, norm_weight_activated), 2)
        if self.fp16:
            logits = logits.float()
        logits = npu_clamp(logits, -1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None: 
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        if self.sample_rate < 1:
            destination["weight"] = self.weight.detach()
        else:
            destination["weight"] = self.weight_activated.data.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.sample_rate < 1:
            self.weight = state_dict["weight"].to(self.weight.device)
            self.weight_mom.zero_()
            self.weight_activated.data.zero_()
            self.weight_activated_mom.zero_()
            self.weight_index.zero_()
        else:
            self.weight_activated.data = state_dict["weight"].to(self.weight_activated.data.device)


class PartialFCAdamW(torch.nn.Module):
    def __init__(self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFCAdamW, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0
        self.weight: torch.Tensor
        self.weight_exp_avg: torch.Tensor
        self.weight_exp_avg_sq: torch.Tensor
        self.weight_activated: torch.nn.Parameter
        self.weight_activated_exp_avg: torch.Tensor
        self.weight_activated_exp_avg_sq: torch.Tensor

        self.is_updated: bool = True
        self.init_weight_update: bool = True

        if self.sample_rate < 1:
            self.register_buffer("weight",
                tensor=torch.normal(0, 0.01, (self.num_local, embedding_size)))
            self.register_buffer("weight_exp_avg",
                tensor=torch.zeros_like(self.weight))
            self.register_buffer("weight_exp_avg_sq",
                tensor=torch.zeros_like(self.weight))
            self.register_parameter("weight_activated",
                param=torch.nn.Parameter(torch.empty(0, 0)))
            self.register_buffer("weight_activated_exp_avg",
                tensor=torch.empty(0, 0))
            self.register_buffer("weight_activated_exp_avg_sq",
                tensor=torch.empty(0, 0))
        else:
            self.weight_activated = torch.nn.Parameter(
                torch.normal(0, 0.01, (self.num_local, embedding_size))
            )
        self.step = 0

        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    @torch.no_grad()
    def sample(self, labels, index_positive, optimizer):
        self.step += 1
        positive = torch.unique(labels[index_positive], sorted=True).npu()
        if self.num_sample - positive.size(0) >= 0:
            perm = torch.rand(size=[self.num_local]).npu()
            perm[positive] = 2.0
            index = torch.topk(perm, k=self.num_sample)[1].npu()
            index = index.sort()[0].npu()
        else:
            index = positive
        self.weight_index = index
        labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        self.weight_activated = torch.nn.Parameter(self.weight[self.weight_index])
        self.weight_activated_exp_avg = self.weight_exp_avg[self.weight_index]
        self.weight_activated_exp_avg_sq = self.weight_exp_avg_sq[self.weight_index]

        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            # TODO the params of partial fc must be last in the params list
            optimizer.state.pop(optimizer.param_groups[-1]["params"][0], None)
            optimizer.param_groups[-1]["params"][0] = self.weight_activated
            if hasattr(optimizer._amp_stash, "all_fp32_params"):
                optimizer._amp_stash.all_fp32_params[-1] = self.weight_activated
            optimizer.state[self.weight_activated]["exp_avg"] = self.weight_activated_exp_avg
            optimizer.state[self.weight_activated]["exp_avg_sq"] = self.weight_activated_exp_avg_sq
            optimizer.state[self.weight_activated]["step"] = self.step
        else:
            raise

    @torch.no_grad()
    def update(self):
        """ partial weight to global
        """
        if self.init_weight_update:
            self.init_weight_update = False
            return

        if self.sample_rate < 1:
            self.weight[self.weight_index] = self.weight_activated
            self.weight_exp_avg[self.weight_index] = self.weight_activated_exp_avg
            self.weight_exp_avg_sq[self.weight_index] = self.weight_activated_exp_avg_sq

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).

        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()
        self.update()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            "last batch size do not equal current batch size: {} vs {}".format(
            self.last_batch_size, batch_size))

        device = local_embeddings.device
        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size), device=device)
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size, dtype=torch.long, device=device) for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            self.sample(labels, index_positive, optimizer)

        norm_embeddings = normalize(embeddings)
        norm_weight_activated = normalize(self.weight_activated)
        logits = torch_npu.npu_format_cast(functional.linear(norm_embeddings, norm_weight_activated), 2)
        if self.fp16:
            logits = logits.float()
        logits = npu_clamp(logits, -1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None: 
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        if self.sample_rate < 1:
            destination["weight"] = self.weight.detach()
        else:
            destination["weight"] = self.weight_activated.data.detach()
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.sample_rate < 1:
            self.weight = state_dict["weight"].to(self.weight.device)
            self.weight_exp_avg.zero_()
            self.weight_exp_avg_sq.zero_()
            self.weight_activated.data.zero_()
            self.weight_activated_exp_avg.zero_()
            self.weight_activated_exp_avg_sq.zero_()
        else:
            self.weight_activated.data = state_dict["weight"].to(self.weight_activated.data.device)


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()

        grads_flatten = apex_C.flatten(grad_list)
        distributed.all_reduce(grads_flatten, distributed.ReduceOp.SUM, async_op=False)
        grad_list_tmp = apex_C.unflatten(grads_flatten, grad_list)

        grad_out = grad_list_tmp[rank]
        grad_list[rank].copy_(grad_out)
        grad_out = grad_list[rank]

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply

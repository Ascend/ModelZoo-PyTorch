# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List
from .. import npu_fused_mode

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class FairseqAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")


@register_optimizer("adam", dataclass=FairseqAdamConfig)
class FairseqAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqAdamConfig, params):
        super().__init__(cfg)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(cfg, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if getattr(cfg, "tpu", False):
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError("--fp16-adam-stats is only supported on GPU")
            # on TPUs we use the Adam defined here, since it
            # automatically casts gradients to FP32
            self._optimizer = Adam(params, **self.optimizer_config)
        elif use_fused_adam:
            logger.info("using FusedAdam")
            self._optimizer = fused_adam_cls(
                params, use_fp16_stats=self.cfg.fp16_adam_stats, **self.optimizer_config
            )
        else:
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError(
                    "--fp16-adam-stats is only supported with FusedAdamV1"
                )
            self._optimizer = Adam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


if npu_fused_mode:
    from collections import defaultdict
    from torch_npu.utils import npu_combine_tensors

    class Adam(torch.optim.Optimizer):
        r"""Implements Adam algorithm.

        This implementation is modified from torch.optim.Adam based on:
        `Fixed Weight Decay Regularization in Adam`
        (see https://arxiv.org/abs/1711.05101)

        It has been proposed in `Adam: A Method for Stochastic Optimization`_.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_

        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ
        """

        def __init__(
                self,
                params,
                lr=1e-3,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                amsgrad=False,
        ):
            defaults = dict(
                lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
            )
            super(Adam, self).__init__(params, defaults)

            self._combined_state_indexed_by_group = None
            self._combined_params_indexed_by_group = None
            self._combined_grads_indexed_by_group = None

        @property
        def supports_memory_efficient_fp16(self):
            return True

        @property
        def supports_flat_params(self):
            return True

        @torch.no_grad()
        def step(self, closure=None):
            """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group_index, _ in enumerate(self.param_groups):
                self._group_step(group_index)

            return loss

        def _init_param_state(self, p_data_fp32, state, amsgrad):
            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p_data_fp32)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
            else:
                state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                if amsgrad:
                    state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                        p_data_fp32
                    )

        def _group_step(self, group_index):
            group = self.param_groups[group_index]

            amsgrad = group.get("amsgrad", False)
            beta1, beta2 = group["betas"]

            # combine tensors
            self._maybe_init_combine_states()
            self._maybe_init_combine_params_and_grads()

            for combined_params_state, combined_params, combined_grads in zip(
                    self._combined_state_indexed_by_group[group_index],
                    self._combined_params_indexed_by_group[group_index],
                    self._combined_grads_indexed_by_group[group_index]):

                if combined_params is None or combined_grads is None:
                    continue

                # use grad with float32
                # if combined_grads.dtype in {torch.float16, torch.bfloat16}:
                #    combined_grads = combined_grads.float()
                if combined_grads.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                exp_avg, exp_avg_sq = combined_params_state["exp_avg"], combined_params_state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = combined_params_state["max_exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(combined_grads, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(combined_grads, combined_grads, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                combined_params_state['step'] += 1
                bias_correction1 = 1 - beta1 ** combined_params_state["step"]
                bias_correction2 = 1 - beta2 ** combined_params_state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    combined_params.add_(
                        combined_params, alpha=-group["weight_decay"] * group["lr"]
                    )

                combined_params.addcdiv_(exp_avg, denom, value=-step_size)

        def _combine_group_param_states(self, group_index):
            group = self.param_groups[group_index]
            params = group['params']
            amsgrad = group['amsgrad']

            step_list = [[], []]
            exp_avg_list = [[], []]
            exp_avg_sq_list = [[], []]
            max_exp_avg_sq_list = [[], []]

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                self._init_param_state(p.data, state, amsgrad)

                kind = 0 if hasattr(p, 'expert') else 1
                step_list[kind].append(state['step'])
                exp_avg_list[kind].append(state['exp_avg'])
                exp_avg_sq_list[kind].append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sq_list[kind].append(state['max_exp_avg_sq'])

            combined_state = [None, None]

            for kind in range(len(combined_state)):
                combined_step = 0
                combined_exp_avg = None
                combined_exp_avg_sq = None
                combined_max_exp_avg_sq = None

                if len(step_list[kind]) > 0:
                    combined_step = step_list[kind][0]
                    combined_exp_avg = npu_combine_tensors(exp_avg_list[kind])
                    combined_exp_avg_sq = npu_combine_tensors(exp_avg_sq_list[kind])
                    combined_max_exp_avg_sq = npu_combine_tensors(max_exp_avg_sq_list[kind])
                else:
                    continue

                combined_state[kind] = defaultdict(dict)
                combined_state[kind]['step'] = combined_step
                combined_state[kind]['exp_avg'] = combined_exp_avg
                combined_state[kind]['exp_avg_sq'] = combined_exp_avg_sq
                combined_state[kind]['max_exp_avg_sq'] = combined_max_exp_avg_sq

            return combined_state

        def _check_initialized(self, groups):
            def _one_kind_initialized(pair):
                return pair and (pair[0] is not None or pair[1] is not None)

            return groups and any(map(_one_kind_initialized, groups))

        def _maybe_init_combine_states(self):
            if self._check_initialized(self._combined_state_indexed_by_group):
                return

            self._combined_state_indexed_by_group = list(map(
                self._combine_group_param_states,
                range(len(self.param_groups))))

        def _combine_params_and_grads(self, group_index):
            group = self.param_groups[group_index]
            params = group['params']

            group_params_list = [[], []]
            group_grads_list = [[], []]

            for p in params:
                if p.grad is None:
                    continue
                param_size = p.storage().size()
                grad_size = p.grad.storage().size()

                if param_size != grad_size:
                    p.grad.data = p.grad.data.clone()

                kind = 0 if hasattr(p, 'expert') else 1
                group_params_list[kind].append(p)
                group_grads_list[kind].append(p.grad)

            combined_params = [None, None]
            combined_grads = [None, None]
            for kind in range(len(combined_params)):
                combined_params[kind] = npu_combine_tensors(group_params_list[kind])
                combined_grads[kind] = npu_combine_tensors(group_grads_list[kind])

            return combined_params, combined_grads

        def _maybe_init_combine_params_and_grads(self):
            if self._check_initialized(self._combined_params_indexed_by_group) and \
                    self._check_initialized(self._combined_grads_indexed_by_group):
                return

            self._combined_params_indexed_by_group = []
            self._combined_grads_indexed_by_group = []

            for group_index in range(len(self.param_groups)):
                combined_params, combined_grads = self._combine_params_and_grads(group_index)
                self._combined_params_indexed_by_group.append(combined_params)
                self._combined_grads_indexed_by_group.append(combined_grads)

        @torch.no_grad()
        def zero_grad(self, set_to_none=False):
            if set_to_none:
                raise ValueError(
                    "set_to_none is not supported in fused optimizers")

            self._maybe_init_combine_params_and_grads()

            for grad_pair in self._combined_grads_indexed_by_group:
                for grad in grad_pair:
                    if grad is None:
                        continue
                    grad.zero_()

        @torch.no_grad()
        def clip_grad_norm_(self, params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
            def grad_exists(p):
                return p is not None and getattr(p, "grad", None) is not None

            if isinstance(params, torch.Tensor):
                params = [params]
            params = list(params)
            grads = [
                p.grad for p in params if grad_exists(p) and not hasattr(p, "expert")
            ]
            expert_grads = [
                p.grad for p in params if grad_exists(p) and hasattr(p, "expert")
            ]

            if len(grads) == 0:
                if len(params) > 0:
                    return params[0].new_tensor(0.0)
                else:
                    return torch.tensor(0.0)

            if len(grads) == 1:
                total_norm = torch.norm(grads[0].detach(), p=2, dtype=torch.float32)
            else:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                elif grads[0].device.type == "xla":
                    device = grads[0].device
                else:
                    device = torch.device("cpu")

                combined_grads_without_expert = self._combined_grads_indexed_by_group[0][1]

                if combined_grads_without_expert is None:
                    if len(params) > 0:
                        return params[0].new_tensor(0.0)
                    else:
                        return torch.tensor(0.0)

                total_norm = torch.norm(combined_grads_without_expert, p=2, dtype=torch.float32).to(device)

            if aggregate_norm_fn is not None:
                total_norm = aggregate_norm_fn(total_norm)

            if max_norm > 0:
                max_norm = float(max_norm)
                clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
                for combined_grad in self._combined_grads_indexed_by_group[0]:
                    combined_grad.mul_(clip_coef)

            return total_norm

        @torch.no_grad()
        def multiply_grads(self, c):
            """Multiplies grads by a constant *c*."""
            # combine tensors
            self._maybe_init_combine_states()
            self._maybe_init_combine_params_and_grads()

            device = torch.cuda.current_device()
            factor = c.to(device) if torch.is_tensor(c) else c
            for combined_grad in self._combined_grads_indexed_by_group[0]:
                if combined_grad is not None:
                    combined_grad.mul_(factor)

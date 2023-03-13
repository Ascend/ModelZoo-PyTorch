from typing import Any, Dict, Union

import torch

from transformers import Trainer
from transformers.file_utils import (
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import AdamW
from transformers.trainer_utils import ShardedDDPOption


if is_apex_available():
    from apex import amp


class FunsdTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs


    def _wrap_model(self, model, training=True):
        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level,
                                                   combine_grad=True)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)

        elif is_sagemaker_dp_enabled():
            model = DDP(model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)
        elif self.args.local_rank != -1:
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
            else:
                find_unused_parameters = True
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        return model


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                if self.use_apex:
                    from apex.optimizers import NpuFusedAdamW
                    optimizer_cls = NpuFusedAdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

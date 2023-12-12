# Copyright 2023 Huawei Technologies Co., Ltd
import torch
from torch import nn

import library.train_util as train_util
import library.sdxl_train_util as sdxl_train_util

class SdxlPretrainModels(nn.Module):
    def __init__(self, unet: nn.Module, text_encoder1: nn.Module, text_encoder2: nn.Module,
                 tokenizer1, tokenizer2, weight_dtype):
        super().__init__()
        self.unet = unet
        self.text_encoder1 = text_encoder1
        self.text_encoder2 = text_encoder2
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.weight_dtype = weight_dtype

    def forward(self, args, batch, accelerator, noise_scheduler, latents):
        input_ids1 = batch["input_ids"]
        input_ids2 = batch["input_ids2"]
        with torch.set_grad_enabled(args.train_text_encoder):
            input_ids1 = input_ids1.to(accelerator.device)
            input_ids2 = input_ids2.to(accelerator.device)
            encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                args.max_token_length,
                input_ids1,
                input_ids2,
                self.tokenizer1,
                self.tokenizer2,
                self.text_encoder1,
                self.text_encoder2,
                None if not args.full_fp16 else self.weight_dtype,
            )

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(
            self.weight_dtype)

        # concat embeddings
        vector_embedding = torch.cat([pool2, embs], dim=1).to(self.weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(self.weight_dtype)

        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler,
                                                                                           latents)

        noisy_latents = noisy_latents.to(self.weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        if not args.full_fp16:
            noisy_latents = noisy_latents.to(torch.float32)
            timesteps = timesteps.to(torch.float32)
            text_embedding = text_embedding.to(torch.float32)
            vector_embedding = vector_embedding.to(torch.float32)

        # Predict the noise residual
        with accelerator.autocast():
            noise_pred = self.unet(noisy_latents, timesteps, text_embedding, vector_embedding)

        return noise_pred, noise, timesteps




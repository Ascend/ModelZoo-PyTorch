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

import os
import time
import argparse
from typing import Callable, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from ais_bench.infer.interface import InferSession


class AscendStableDiffusionPipeline(StableDiffusionPipeline):
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        clip_session,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(prompt,
                                     padding="max_length",
                                     max_length=self.tokenizer.model_max_length,
                                     truncation=True,
                                     return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            print("[warning] The following part of your input was truncated"
                  " because CLIP can only handle sequences up to"
                  f" {self.tokenizer.model_max_length} tokens: {removed_text}")

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = clip_session([text_input_ids.numpy()])
        text_embeddings = [torch.from_numpy(text) for text in text_embeddings]

        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                                f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                                 f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                                 " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, 
                                          padding="max_length",
                                          max_length=max_length, 
                                          truncation=True, 
                                          return_tensors="pt")

            if (hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def ascend_infer(
        self,
        prompt: Union[str, List[str]],
        clip_session,
        unet_session,
        vae_session,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (畏) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt,
                                              device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt,
                                              clip_session)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents,
                                       height,
                                       width,
                                       text_embeddings.dtype,
                                       device,
                                       generator,
                                       latents)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            # noise_pred = self.unet(latent_model_input, t[None], encoder_hidden_states=text_embeddings).sample
            noise_pred = torch.from_numpy(
                unet_session(
                    [
                        latent_model_input.numpy(),
                        t[None].numpy(),
                        text_embeddings.numpy(),
                    ]
                )[0]
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        # image = self.decode_latents(latents)

        latents = 1 / self.vae.config.scaling_factor * latents

        latents = self.vae.post_quant_conv(latents)
        image = torch.from_numpy(vae_session([latents.numpy()])[0])

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, 
                                                          device, 
                                                          text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return (image, has_nsfw_concept)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="A text file of prompts for generating images.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Base path of om models.",
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./results", 
        help="Path to save result images.",
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50, 
        help="Number of inference steps.",
    )
    parser.add_argument("--device", type=int, default=0, help="Device id.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir

    pipe = AscendStableDiffusionPipeline.from_pretrained(args.model).to("cpu")

    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        prompts = [line.strip() for line in f]

    clip_om = os.path.join(args.model_dir, "clip", "clip.om")
    unet_om = os.path.join(args.model_dir, "unet", "unet.om")
    vae_om = os.path.join(args.model_dir, "vae", "vae.om")

    clip_session = InferSession(args.device, clip_om).infer
    unet_session = InferSession(args.device, unet_om).infer
    vae_session = InferSession(args.device, vae_om).infer

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o744)

    infer_num = len(prompts)
    start_time = time.time()
    for i, prompt in enumerate(prompts):
        image = pipe.ascend_infer(
            prompt,
            clip_session,
            unet_session,
            vae_session,
            num_inference_steps=args.steps,
        )
        image = image[0][0]
        image.save(os.path.join(save_dir, f"illustration_{i}.png"))

    use_time = time.time() - start_time
    print(
        f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
        f"average time: {use_time/infer_num:.3f}s"
    )


if __name__ == "__main__":
    main()

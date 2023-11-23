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
import stat
import time
import argparse
import logging
import json
import csv
from typing import Callable, List, Optional, Union
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import ascendie as aie
from background_runtime_dynamic import BackgroundRuntime


class PromptLoader:
    def __init__(
        self,
        prompt_file: str,
        prompt_file_type: str,
        batch_size: int,
        num_images_per_prompt: int = 1,
    ):
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file)

        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file)

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    def load_prompts_plain(self, file_path: str):
        with os.fdopen(os.open(file_path, os.O_RDONLY, stat.S_IRUSR), "r") as f:
            for i, line in enumerate(f):
                prompt = line.strip()
                self.prompts.append((prompt, 0))
                
    def load_prompts_parti(self, file_path: str):
        with os.fdopen(os.open(file_path, os.O_RDONLY, stat.S_IRUSR), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))


class AIEStableDiffusionPipeline(StableDiffusionPipeline):
    device_0 = None
    device_1 = None
    runtime = None
    engines = {}
    contexts = {}
    buffer_bindings = {}
    use_parallel_inferencing = False
    unet_bg = None
    use_dynamic_dims = False
    resolution = []
    
    def parse_onnx_model(self, onnx_path, model = "normal"):
        builder = aie.Builder.create_builder(b'Ascend310P3')
        logging.info("finish create builder")
        network = builder.create_network()
        logging.info("finish create network")
        parser = aie.OnnxModelParser()
        if not parser.parse_model(network, onnx_path):
            logging.error("parse false")
        logging.info("finish parse network")
        builder_config = aie.BuilderConfig()
        if (model == "unetdy" and self.device_1 is None):
            profile = aie.DynamicProfile()
            profile.add_dims({"latent_model_input": aie.Dims([2, 4, 64, 64]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([2, 77, 768])})
            profile.add_dims({"latent_model_input": aie.Dims([2, 4, 72, 96]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([2, 77, 768])})
            profile.add_dims({"latent_model_input": aie.Dims([2, 4, 96, 72]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([2, 77, 768])})
            builder_config.dyn_profile = profile
        elif (model == "unetdy" and self.device_1 is not None):
            profile = aie.DynamicProfile()
            profile.add_dims({"latent_model_input": aie.Dims([1, 4, 64, 64]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([1, 77, 768])})
            profile.add_dims({"latent_model_input": aie.Dims([1, 4, 72, 96]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([1, 77, 768])})
            profile.add_dims({"latent_model_input": aie.Dims([1, 4, 96, 72]), "t": aie.Dims([1]),
                "encoder_hidden_states": aie.Dims([1, 77, 768])})
            builder_config.dyn_profile = profile
        if (model == "vaedy"):
            profile = aie.DynamicProfile()
            profile.add_dims({"latents": aie.Dims([1, 4, 64, 64])})
            profile.add_dims({"latents": aie.Dims([1, 4, 72, 96])})
            profile.add_dims({"latents": aie.Dims([1, 4, 96, 72])})
            builder_config.dyn_profile = profile
        model_data = builder.build_model(network, builder_config)
        logging.info("finish build model")
        if not model_data:
            logging.error("build model failed")
        with open(onnx_path[:-5] + '_aie.om', 'wb') as f:
            f.write(model_data.data)
        del builder
        return model_data
    
    def build_engines_onnx(self, clip_onnx_path, unet_onnx_path, vae_onnx_path):
        ret = aie.set_device(self.device_0)
        self.runtime = aie.Runtime.get_instance()
        clip_model_data = self.parse_onnx_model(clip_onnx_path)
        engine1 = self.runtime.deserialize_engine_from_mem(clip_model_data)
        context1 = engine1.create_context()
        self.engines['clip'] = engine1
        self.contexts['clip'] = context1
        unet_model_data = self.parse_onnx_model(unet_onnx_path, "unetdy") if self.use_dynamic_dims else \
            self.parse_onnx_model(unet_onnx_path)
        engine2 = self.runtime.deserialize_engine_from_mem(unet_model_data)
        context2 = engine2.create_context()
        self.engines['unet'] = engine2
        self.contexts['unet'] = context2
        vae_model_data = self.parse_onnx_model(vae_onnx_path, "vaedy") if self.use_dynamic_dims else \
            self.parse_onnx_model(vae_onnx_path)
        engine3 = self.runtime.deserialize_engine_from_mem(vae_model_data)
        context3 = engine3.create_context()
        self.engines['vae'] = engine3
        self.contexts['vae'] = context3
        if self.device_1 is not None:
            self.unet_bg = BackgroundRuntime.clone(self.device_1, unet_onnx_path, self.engines['unet'],
                                                   self.resolution)
            self.use_parallel_inferencing = True
        
    def build_engines(self, clip_path, unet_path, vae_path):
        ret = aie.set_device(self.device_0)
        self.runtime = aie.Runtime.get_instance()
        engine1 = self.runtime.deserialize_engine_from_file(clip_path)
        context1 = engine1.create_context()
        self.engines['clip'] = engine1
        self.contexts['clip'] = context1
        engine2 = self.runtime.deserialize_engine_from_file(unet_path)
        context2 = engine2.create_context()
        self.engines['unet'] = engine2
        self.contexts['unet'] = context2
        engine3 = self.runtime.deserialize_engine_from_file(vae_path)
        context3 = engine3.create_context()
        self.engines['vae'] = engine3
        self.contexts['vae'] = context3
        if self.device_1 is not None:
            self.unet_bg = BackgroundRuntime.clone(self.device_1, unet_path,
                                                   self.engines['unet'], self.resolution)
            self.use_parallel_inferencing = True

    def malloc_io_binding(self):
        buffer_binding1 = aie.IO_binding(self.engines['clip'],
                                    self.contexts['clip'])
        buffer_binding2 = aie.IO_binding(self.engines['unet'],
                                    self.contexts['unet'])
        buffer_binding3 = aie.IO_binding(self.engines['vae'], self.contexts['vae'])
        self.buffer_bindings['clip'] = buffer_binding1
        self.buffer_bindings['unet'] = buffer_binding2
        self.buffer_bindings['vae'] = buffer_binding3

    def release(self):
        aie.release_IO_binding(self.buffer_bindings['clip'])
        aie.release_IO_binding(self.buffer_bindings['unet'])
        aie.release_IO_binding(self.buffer_bindings['vae'])

    @torch.no_grad()
    def ascendie_infer(
        self,
        prompt: Union[str, List[str]],
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
        callback: Optional[Callable[[int, int, torch.FloatTensor],
                                    None]] = None,
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
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Only applies to `schedulers.DDIMScheduler`, will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A torch generator to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between `PIL.Image.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
 
        Returns:
            `tuple`:
            the first element is a list with the generated images, and the second element is a
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
        # of the Imagen paper. `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        text_embeddings_dtype = text_embeddings.dtype

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents, height, width,
                                       text_embeddings_dtype, device,
                                       generator, latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if self.use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            text_embeddings, text_embeddings_2 = text_embeddings.chunk(2)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            if not self.use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            if self.use_parallel_inferencing and do_classifier_free_guidance:
                self.unet_bg.infer_asyn([
                    latent_model_input.numpy(),
                    t[None].numpy().astype(np.int32),
                    text_embeddings_2.numpy(),
                ])

            input_data_unet = [
                latent_model_input.numpy(),
                t[None].numpy().astype(np.int32),
                text_embeddings.numpy()
            ]
            output_data = aie.execute(input_data_unet, self.buffer_bindings['unet'],
                                  self.contexts['unet'])[0]
            unetH = height//8
            unetW = width//8
            if self.use_parallel_inferencing:
                noise_pred = torch.from_numpy(
                    np.frombuffer(output_data, dtype=np.float32)[:4 * unetH * unetW].reshape(
                        (1, 4, unetH, unetW)))
            else:
                noise_pred = torch.from_numpy(
                    np.frombuffer(output_data, dtype=np.float32)[:2 * 4 * unetH * unetW].reshape(
                        (2, 4, unetH, unetW)))

            # perform guidance
            if do_classifier_free_guidance:
                if self.use_parallel_inferencing:
                    noise_pred_text = torch.from_numpy(
                        self.unet_bg.wait_and_get_outputs()[0])
                else:
                    noise_pred, noise_pred_text = noise_pred.chunk(2)

                noise_pred = noise_pred + guidance_scale * (noise_pred_text -
                                                            noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        latents = 1 / self.vae.config.scaling_factor * latents

        latents = self.vae.post_quant_conv(latents)

        # run inference
        input_data_vae = [latents.numpy()]
        image_buffer = aie.execute(input_data_vae, self.buffer_bindings['vae'],
                               self.contexts['vae'])[0]
        image = torch.from_numpy(
            np.frombuffer(image_buffer, dtype=np.float32)[:3 * height * width].reshape(
                (1, 3, height, width)))

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings_dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return (image, has_nsfw_concept)
    
    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
 
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt,
                                         padding="max_length",
                                         return_tensors="pt").input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            print("[warning] The following part of your input was truncated"
                  " because CLIP can only handle sequences up to"
                  f" {self.tokenizer.model_max_length} tokens: {removed_text}")

        # run inference
        input_data_np = [text_input_ids.numpy()]
        output_data = aie.execute(input_data_np, self.buffer_bindings['clip'],
                              self.contexts['clip'])[0]
        # SD1.5
        text_embeddings = torch.from_numpy(
            np.frombuffer(output_data, dtype=np.float32).reshape((1, 77, 768)))

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
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

            # run inference
            uncond_input_data_np = [uncond_input.input_ids.numpy()]
            uncond_output_data = aie.execute(uncond_input_data_np,
                                         self.buffer_bindings['clip'],
                                         self.contexts['clip'])[0]

            # SD1.5
            uncond_embeddings = torch.from_numpy(
                np.frombuffer(uncond_output_data, dtype=np.float32).reshape(
                    (1, 77, 768)))

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [int(v) for v in value.split(',')]
        for ivalue in ilist[:2]:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError(
                    "{} of device:{} is invalid. valid value range is [{}, {}]"
                    .format(ivalue, value, min_value, max_value))
        return ilist[:2]
    else:
        # default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError(
                "device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, min_value, max_value))
        return ivalue


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
        default="prompts.txt",
        help="A prompt file used to generate images.",
    )
    parser.add_argument(
        "--prompt_file_type", 
        choices=["plain", "parti"],
        default="plain", 
        help="Type of prompt file.",
    )
    parser.add_argument(
        "--om_model_dir",
        type=str,
        default="./om_models",
        help="Base path of om models.",
    )
    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        default="./onnx_models",
        help="Base path of onnx models.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images.",
    )
    parser.add_argument(
        "--info_file_save_path", 
        type=str, 
        default="./image_info.json", 
        help="Path to save image information file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--device",
        type=check_device_range_valid,
        default=0,
        help="NPU device id. Give 2 ids to enable parallel inferencing."
    )
    parser.add_argument(
        "--use_onnx_parser",
        action="store_true",
        help="if use onnx parser."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="Number of images generated for each prompt.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size."
    )
    parser.add_argument(
        "--use_dynamic_dims",
        action="store_true",
        help="if use dynamic dims."
    )
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="512,512", 
        help="The resolution of image.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir

    pipe = AIEStableDiffusionPipeline.from_pretrained(args.model).to("cpu")
    if isinstance(args.device, list):
        pipe.device_0, pipe.device_1 = args.device
    else:
        pipe.device_0 = args.device
    pipe.use_dynamic_dims = args.use_dynamic_dims
    resolution = args.resolution.strip().split(",")
    resolution = list(map(int, resolution))
    pipe.resolution = resolution
    if args.use_onnx_parser:
        clip_onnx_path = os.path.join(args.onnx_model_dir, "clip", "clip.onnx")
        unet_onnx_path = os.path.join(args.onnx_model_dir, "unetdy", "unetdy.onnx")
        vae_onnx_path = os.path.join(args.onnx_model_dir, "vae", "vaedy.onnx")
        pipe.build_engines_onnx(clip_onnx_path, unet_onnx_path, vae_onnx_path)
    else:
        clip_om_path = os.path.join(args.om_model_dir, "clip", "clip.om")
        unet_om_path = os.path.join(args.om_model_dir, "unet", "unet.om")
        vae_om_path = os.path.join(args.om_model_dir, "vae", "vae.om")
        pipe.build_engines(clip_om_path, unet_om_path, vae_om_path)
    pipe.malloc_io_binding()

    use_time = 0
    prompt_loader = PromptLoader(args.prompt_file, 
                                 args.prompt_file_type, 
                                 args.batch_size,
                                 args.num_images_per_prompt,
                                 )
    infer_num = 0
    image_info = []
    current_prompt = None
    for i, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']
        
        print(f"[{infer_num + n_prompts}/{len(prompt_loader)}]: {prompts}")
        infer_num += args.batch_size

        resolution = args.resolution.strip().split(",")
        height = int(resolution[0])
        width = int(resolution[1])
        start_time = time.time()
        images = pipe.ascendie_infer(
            prompts,
            height,
            width,
            num_inference_steps=args.steps,
        ) if args.use_dynamic_dims else pipe.ascendie_infer(
            prompts,
            num_inference_steps=args.steps,
        )
        use_time += time.time() - start_time
        for j in range(n_prompts):
            image_save_path = os.path.join(save_dir, f"{save_names[j]}.png")
            image = images[0][j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
          f"average time: {use_time/infer_num:.3f}s")
    if pipe.unet_bg:
        pipe.unet_bg.stop()

    pipe.release()
    
    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)
        
    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o644), "w") as f:
        json.dump(image_info, f)


if __name__ == "__main__":
    main()

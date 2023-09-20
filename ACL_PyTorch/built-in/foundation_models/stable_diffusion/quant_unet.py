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
import argparse
from typing import Callable, List, Optional, Union

import onnx
import torch
import numpy as np
from ais_bench.infer.interface import InferSession
from modelslim.onnx.squant_ptq.onnx_quant_tools import OnnxCalibrator
from modelslim.onnx.squant_ptq.quant_config import QuantConfig
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler

from background_session import BackgroundInferSession
from pipeline_ascend_stable_diffusion import AscendStableDiffusionPipeline
from stable_diffusion_ascend_infer import check_device_range_valid


class StableDiffusionDumpPipeline(AscendStableDiffusionPipeline):
    @torch.no_grad()
    def dump_data(
        self,
        prompt: Union[str, List[str]],
        clip_session: InferSession,
        unet_sessions: list,
        dump_num: int = 10,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt,
                                              clip_session)

        text_embeddings_dtype = text_embeddings.dtype

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt,
                                       num_channels_latents,
                                       height,
                                       width,
                                       text_embeddings_dtype,
                                       device,
                                       generator,
                                       latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        unet_session, unet_session_bg = unet_sessions
        use_parallel_inferencing = unet_session_bg is not None
        if use_parallel_inferencing and do_classifier_free_guidance:
            # Split embeddings
            text_embeddings, text_embeddings_2 = text_embeddings.chunk(2)
            text_embeddings_2 = text_embeddings_2.numpy()

        text_embeddings = text_embeddings.numpy()

        dump_data = []
        start_id = num_inference_steps // 2 - dump_num // 2
        end_id = start_id + dump_num

        for i, t in enumerate(self.progress_bar(timesteps)):
            t_numpy = t[None].numpy()

            # expand the latents if we are doing classifier free guidance
            if not use_parallel_inferencing and do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).numpy()
            if start_id <= i < end_id:
                dump_data.append([latent_model_input, t_numpy, text_embeddings])

            # predict the noise residual
            if use_parallel_inferencing and do_classifier_free_guidance:
                unet_session_bg.infer_asyn(
                    [
                        latent_model_input,
                        t_numpy,
                        text_embeddings_2,
                    ]
                )

            noise_pred = torch.from_numpy(
                unet_session.infer(
                    [
                        latent_model_input,
                        t_numpy,
                        text_embeddings,
                    ]
                )[0]
            )

            # perform guidance
            if do_classifier_free_guidance:
                if use_parallel_inferencing:
                    noise_pred_text = torch.from_numpy(unet_session_bg.wait_and_get_outputs()[0])
                else:
                    noise_pred, noise_pred_text = noise_pred.chunk(2)
 
                noise_pred = noise_pred + guidance_scale * (noise_pred_text - noise_pred)
 
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample
 
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        return dump_data


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
        "--model_dir",
        type=str,
        default="./models",
        help="Base path of om models.",
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="unet_quant", 
        help="Path to save result images.",
    )
    parser.add_argument(
        "--scheduler", 
        choices=["DDIM", "Euler", "DPM"],
        default="DDIM", 
        help="Type of Sampling methods. Can choose from DDIM, Euler, DPM",
    )
    parser.add_argument(
        "--device", 
        type=check_device_range_valid, 
        default=0, 
        help="NPU device id. Give 2 ids to enable parallel inferencing."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50, 
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--data_num", 
        type=int, 
        default=10,
        help="the number of real data used in quant process"
    )
    parser.add_argument(
        "--data_free", 
        action='store_true', 
        help="do not use real data"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    unet_onnx = os.path.join(args.model_dir, "unet", "unet.onnx")

    if args.data_free:
        data = [[]]

    input_shape = ''
    model = onnx.load(unet_onnx)
    inputs = model.graph.input

    for inp in inputs:
        dims = inp.type.tensor_type.shape.dim
        shape = [str(x.dim_value) for x in dims]
        input_shape += inp.name + ':' + ','.join(shape) + ';'
        if args.data_free:
            dtype = inp.type.tensor_type.elem_type
            size = [x.dim_value for x in dims]
            if dtype == 1:
                data[0].append(np.random.random(size).astype(np.float32))
            if dtype == 7:
                data[0].append(np.random.randint(10, size).astype(np.int64))

    if not args.data_free:
        device = None
        device_2 = None

        if isinstance(args.device, list):
            device, device_2 = args.device
        else:
            device = args.device
        
        batch_size = inputs[0].type.tensor_type.shape.dim[0].dim_value
        if not device_2:
            batch_size = batch_size // 2

        pipe = StableDiffusionDumpPipeline.from_pretrained(args.model).to("cpu")

        if args.scheduler == "DDIM":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if args.scheduler == "Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        if args.scheduler == "DPM":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        clip_om = os.path.join(args.model_dir, "clip", "clip.om")
        unet_om = os.path.join(args.model_dir, "unet", "unet.om")

        clip_session = InferSession(device, clip_om)
        unet_session = InferSession(device, unet_om)

        unet_session_bg = None
        if device_2:
            unet_session_bg = BackgroundInferSession.clone(unet_session, device_2)

        with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
            prompts = [line.strip() for line in f]

        data = pipe.dump_data(
            prompts[:batch_size],
            clip_session,
            [unet_session, unet_session_bg],
            args.data_num,
            num_inference_steps=args.steps
        )

        if unet_session_bg:
            unet_session_bg.stop()
    
    config = QuantConfig(
        disable_names=[],
        quant_mode=0,
        amp_num=0,
        use_onnx=False,
        disable_first_layer=False,
        quant_param_ops=['Conv'],
        atc_input_shape=input_shape[:-1],
        num_input=len(inputs)
    )

    calib = OnnxCalibrator(unet_onnx, config, calib_data=data)
    calib.run()
    quant_path = os.path.join(args.model_dir, args.save_path)
    if not os.path.exists(quant_path):
        os.makedirs(quant_path, mode=0o744)
    quant_onnx = os.path.join(quant_path, 'unet.onnx')
    calib.export_quant_onnx(quant_onnx, use_external=True)

if __name__ == "__main__":
    main()

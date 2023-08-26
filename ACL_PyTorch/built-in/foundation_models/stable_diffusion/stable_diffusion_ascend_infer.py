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

from ais_bench.infer.interface import InferSession
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler

from background_session import BackgroundInferSession, SessionIOInfo
from pipeline_ascend_stable_diffusion import AscendStableDiffusionPipeline


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [ int(v) for v in value.split(',') ]
        for ivalue in ilist[:2]:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("{} of device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, value, min_value, max_value))
        return ilist[:2]
    else:
		# default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
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

    return parser.parse_args()


def main():
    args = parse_arguments()
    save_dir = args.save_dir
    device = None
    device_2 = None

    if isinstance(args.device, list):
        device, device_2 = args.device
    else:
        device = args.device

    pipe = AscendStableDiffusionPipeline.from_pretrained(args.model).to("cpu")

    if args.scheduler == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if args.scheduler == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        prompts = [line.strip() for line in f]

    clip_om = os.path.join(args.model_dir, "clip", "clip.om")
    vae_om = os.path.join(args.model_dir, "vae", "vae.om")

    if device_2:
        unet_om = os.path.join(args.model_dir, "unet", "unet_bs1.om")
    else:
        unet_om = os.path.join(args.model_dir, "unet", "unet_bs2.om")

    clip_session = InferSession(device, clip_om)
    vae_session = InferSession(device, vae_om)
    unet_session = InferSession(device, unet_om)

    unet_session_bg = None
    if device_2:
        unet_session_bg = BackgroundInferSession.clone(unet_session, device_2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o744)

    infer_num = len(prompts)
    use_time = 0
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        image = pipe.ascend_infer(
            prompt,
            clip_session,
            [unet_session, unet_session_bg],
            vae_session,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
        )
        use_time += time.time() - start_time
        image = image[0][0]
        image.save(os.path.join(save_dir, f"illustration_{i}.png"))

    if unet_session_bg:
        unet_session_bg.stop()

    print(
        f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
        f"average time: {use_time/infer_num:.3f}s"
    )


if __name__ == "__main__":
    main()
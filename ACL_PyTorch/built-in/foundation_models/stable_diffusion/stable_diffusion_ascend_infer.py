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
import csv
import time
import json
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
        help="A prompt file used to generate images.",
    )
    parser.add_argument(
        "--prompt_file_type", 
        choices=["normal", "parti"],
        default="normal", 
        help="Type of prompt file.",
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
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="Number of images generated for each prompt.",
    )
    parser.add_argument(
        "--max_num_prompts",
        default=0,
        type=int,
        help="Limit the number of prompts (0: no limit).",
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

    prompts = []
    catagories = []
    with os.fdopen(os.open(args.prompt_file, os.O_RDONLY), "r") as f:
        if args.prompt_file_type == "normal":
            prompts = [line.strip() for line in f]
            if args.max_num_prompts > 0:
                prompts = prompts[:args.max_num_prompts]

            catagories = ["Not_specified"] * len(prompts)

        elif args.prompt_file_type == "parti":
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if args.max_num_prompts > 0 and i == args.max_num_prompts:
                    break

                prompts.append(line[0])
                catagories.append(line[1])

    clip_om = os.path.join(args.model_dir, "clip", "clip.om")
    vae_om = os.path.join(args.model_dir, "vae", "vae.om")
    unet_om = os.path.join(args.model_dir, "unet", "unet.om")

    clip_session = InferSession(device, clip_om)
    vae_session = InferSession(device, vae_om)
    unet_session = InferSession(device, unet_om)

    unet_session_bg = None
    if device_2:
        unet_session_bg = BackgroundInferSession.clone(unet_session, device_2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o744)

    infer_num = len(prompts) * args.num_images_per_prompt
    image_info = []
    use_time = 0
    for i, (prompt, category) in enumerate(zip(prompts, catagories)):
        images = []
        print(f"[{i+1}/{len(prompts)}]: {prompt}")
        for j in range(args.num_images_per_prompt):
            image_save_path = os.path.join(save_dir, f"{i}_{j}.png")
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
            image.save(image_save_path)
            images.append(image_save_path)

        image_info.append({'images': images, 'prompt': prompt, 'category': category})

    if unet_session_bg:
        unet_session_bg.stop()

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)
        
    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR|os.O_CREAT, 0o644), "w") as f:
        json.dump(image_info, f)

    print(
        f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
        f"average time: {use_time/infer_num:.3f}s"
    )


if __name__ == "__main__":
    main()
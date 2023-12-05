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

import argparse 
import os
import random
import time

from PIL import Image
from ais_bench.infer.interface import InferSession
import cv2
import einops
import numpy as np
from pytorch_lightning import seed_everything
import torch

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./models/control_sd15_canny.pth",
        help="Path or name of the pre-trained model."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./test_imgs/dog.png",
        help="Path or name of the image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Prompt",
        help="label=Prompt"
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default="best quality, extremely detailed",
        help="added prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default="longbody, lowres, bad anatomy, bad hands, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality",
        help="negative prompt"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="image_num"
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        help="image resolution"
    )
    parser.add_argument(
        "--guess_mode",
        type=bool,
        default=False,
        help="guess mode"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="control strength"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="guidance scale"
    ) 
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="seed"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta"
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="canny low threshold"
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="canny high threshold"
    )
    parser.add_argument(
        "--control_model_dir",
        type=str,
        default="./models",
        help="Base path of om models "
    )
    parser.add_argument(
        "--sd_model_dir",
        type=str,
        default="./models",
        help="Base path of om models."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images."
    )       
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id."
    )

    return parser.parse_args()


def process(model, ddim_sampler, sd_session, control_session, input_image, 
            prompt, a_prompt, n_prompt, num_samples, image_resolution, 
            ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, 
            high_threshold,
            ):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        apply_canny = CannyDetector()
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map= HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(1)],dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()


        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": 
                [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], 
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [
        	strength * (0.825 ** float(12 - i)) for i in range(13)] 
        	if guess_mode else ([strength] * 13
        )

        samples, intermediates = ddim_sampler.sample(
        	ddim_steps, num_samples,                  
            shape, sd_session, control_session,
            cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
        	einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + \
        	127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    return [255 - detected_map] + results


def main():
    args = parse_arguments()

    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(args.model, location="cpu"))
    model = model.cpu()

    ddim_sampler = DDIMSampler(model)

    sd_om = args.sd_model_dir
    control_om = args.control_model_dir

    sd_session = InferSession(args.device, sd_om)
    control_session = InferSession(args.device, control_om)

    input_image = cv2.imread(args.image)
    output = process(model, ddim_sampler, sd_session, control_session, input_image, 
                     args.prompt, args.a_prompt, args.n_prompt, args.num_samples,
                     args.image_resolution, args.ddim_steps, args.guess_mode,
                     args.strength, args.scale, args.seed, args.eta, args.low_threshold,
                     args.high_threshold)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, mode=0o744)
    img0 = Image.fromarray(output[0])
    img1 = Image.fromarray(output[1])
    img0.save(os.path.join(args.save_dir, "cannyimg.png"))
    img1.save(os.path.join(args.save_dir, "diffusionimg.png"))
    

if __name__=="__main__":
    main()
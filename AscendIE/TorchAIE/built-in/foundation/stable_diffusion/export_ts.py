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
from argparse import Namespace

import torch
from diffusers import StableDiffusionPipeline


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./",
        help="Path of directory to save ONNX models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./stable-diffusion-v1-5",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )

    return parser.parse_args()


class Clipexport(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model(x)[0]


def export_clip(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the text encoder...")
    clip_path = os.path.join(save_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o640)

    clip_model = sd_pipeline.text_encoder

    max_position_embeddings = clip_model.config.max_position_embeddings
    print(f'max_position_embeddings: {max_position_embeddings}')
    dummy_input = torch.ones([batch_size, max_position_embeddings], dtype=torch.int64)
    clip_export = Clipexport(clip_model)

    torch.jit.trace(clip_export, dummy_input).save(os.path.join(clip_path, "clip.pt"))


class Unetexport(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet_model(sample, timestep, encoder_hidden_states)[0]


def export_unet(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o640)

    unet_model = sd_pipeline.unet
    clip_model = sd_pipeline.text_encoder

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size = clip_model.config.hidden_size
    max_position_embeddings = clip_model.config.max_position_embeddings
    dummy_input = (
        torch.ones([batch_size, in_channels, sample_size, sample_size], dtype=torch.float32),
        torch.ones([1], dtype=torch.int64),
        torch.ones(
            [batch_size, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
        ),
    )

    unet = Unetexport(unet_model)
    unet.eval()

    torch.jit.trace(unet, dummy_input).save(os.path.join(unet_path, f"unet_bs{batch_size}.pt"))


class VaeExport(torch.nn.Module):
    def __init__(self, vae_model):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents):
        return self.vae_model.decode(latents)[0]


def export_vae(sd_pipeline: StableDiffusionPipeline, save_dir: str, batch_size: int) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o640)

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    dummy_input = torch.ones([batch_size, in_channels, sample_size, sample_size])
    vae_export = VaeExport(vae_model)
    torch.jit.trace(vae_export, dummy_input).save(os.path.join(vae_path, "vae.pt"))


def export_onnx(model_path: str, save_dir: str, batch_size: int) -> None:
    pipeline = StableDiffusionPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir, batch_size)

    export_unet(pipeline, save_dir, batch_size)
    export_unet(pipeline, save_dir, batch_size * 2)

    export_vae(pipeline, save_dir, batch_size)


def main():
    args = parse_arguments()
    export_onnx(args.model, args.output_dir, args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()

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
from argparse import Namespace

import torch
from diffusers import StableDiffusionPipeline


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./models",
        help="Path of directory to save ONNX models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path or name of the pre-trained model.",
    )

    return parser.parse_args()


def export_clip(sd_pipeline: StableDiffusionPipeline, save_dir: str) -> None:
    print("Exporting the text encoder...")
    clip_path = os.path.join(save_dir, "clip")
    if not os.path.exists(clip_path):
        os.makedirs(clip_path, mode=0o744)

    clip_model = sd_pipeline.text_encoder

    max_position_embeddings = clip_model.config.max_position_embeddings
    dummy_input = torch.ones([1, max_position_embeddings], dtype=torch.int64)

    torch.onnx.export(
        clip_model,
        dummy_input,
        os.path.join(clip_path, "clip.onnx"),
        input_names=["prompt"],
        output_names=["text_embeddings"],
        opset_version=11,
    )


def export_unet(sd_pipeline: StableDiffusionPipeline, save_dir: str) -> None:
    print("Exporting the image information creater...")
    unet_path = os.path.join(save_dir, "unet")
    if not os.path.exists(unet_path):
        os.makedirs(unet_path, mode=0o744)

    unet_model = sd_pipeline.unet
    clip_model = sd_pipeline.text_encoder

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.in_channels
    encoder_hidden_size = clip_model.config.hidden_size
    max_position_embeddings = clip_model.config.max_position_embeddings

    dummy_input = (
        torch.ones([2, in_channels, sample_size, sample_size], dtype=torch.float32),
        torch.ones([1], dtype=torch.int64),
        torch.ones(
            [2, max_position_embeddings, encoder_hidden_size], dtype=torch.float32
        ),
    )

    torch.onnx.export(
        unet_model,
        dummy_input,
        os.path.join(unet_path, "unet.onnx"),
        input_names=["latent_model_input", "t", "encoder_hidden_states"],
        output_names=["sample"],
        opset_version=11,
    )


def export_vae(sd_pipeline: StableDiffusionPipeline, save_dir: str) -> None:
    print("Exporting the image decoder...")

    vae_path = os.path.join(save_dir, "vae")
    if not os.path.exists(vae_path):
        os.makedirs(vae_path, mode=0o744)

    vae_model = sd_pipeline.vae
    unet_model = sd_pipeline.unet

    sample_size = unet_model.config.sample_size
    in_channels = unet_model.config.out_channels

    dummy_input = torch.ones([1, in_channels, sample_size, sample_size])

    torch.onnx.export(
        vae_model.decoder,
        dummy_input,
        os.path.join(vae_path, "vae.onnx"),
        input_names=["latents"],
        output_names=["image"],
        opset_version=11,
    )


def export_onnx(model_path: str, save_dir: str) -> None:
    pipeline = StableDiffusionPipeline.from_pretrained(model_path).to("cpu")

    export_clip(pipeline, save_dir)
    export_unet(pipeline, save_dir)
    export_vae(pipeline, save_dir)

    print("Done.")


def main():
    args = parse_arguments()
    export_onnx(args.model, args.output_dir)


if __name__ == "__main__":
    main()

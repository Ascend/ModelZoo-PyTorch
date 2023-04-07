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
import torch
import shutil

from diffusers import StableDiffusionPipeline


def main():
    clip_path = "./models/clip"
    unet_path = "./models/unet"
    vae_path = "./models/vae"
    
    if os.path.exists("./models"):
        shutil.rmtree("./models")
    
    os.makedirs(clip_path)
    os.makedirs(unet_path)
    os.makedirs(vae_path)

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")
    
    # export clip
    torch.onnx.export(
        pipe.text_encoder,
        (torch.ones([1, 77], dtype = torch.int64)),
        f"{clip_path}/clip.onnx",
        input_names=["prompt"],
        output_names=["text_embeddings"],
        opset_version=11,
        verbose=False,
    )
    print("[info] export clip onnx success!")
    
    # export unet
    torch.onnx.export(
        pipe.unet,
        (
            torch.ones([2, 4, 64, 64], dtype = torch.float32),
            torch.ones([1], dtype = torch.int64),
            torch.ones([2, 77, 768], dtype = torch.float32)
        ),
        f"{unet_path}/unet.onnx",
        input_names=["latent_model_input", "t", "encoder_hidden_states"],
        output_names=["sample"],
        opset_version=11,
        verbose=False,
    )
    print("[info] export unet onnx success!")
    
    # export vae
    torch.onnx.export(
        pipe.vae.decoder,
        (torch.ones([1, 4, 64, 64])),
        f"{vae_path}/vae.onnx",
        input_names=["latents"],
        output_names=["image"],
        opset_version=11,
        verbose=False,
    )
    print("[info] export vae onnx success!")


if __name__ == '__main__':
    main()

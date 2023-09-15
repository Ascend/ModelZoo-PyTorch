# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import clip
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torchvision.transforms as T


class ImgModelWrapper(nn.Module):
    def __init__(self, query_len=10, query_layer=31, v_embed_dim=768,
                 v_depth=8, v_num_heads=16, v_mlp_ratio=4.0):
        super(ImgModelWrapper, self).__init__()
        self.clip, _ = clip.load('ViT-L/14', download_root='./')
        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # define visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, 4096)
        self.visual_proj_norm = nn.LayerNorm(4096)
    
    def clip_encode_image(self, x):
        # modified form Clip
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1],
                                   dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        
        return x
    
    def forward(self, imgs):
        img = imgs.transpose(1, 2).transpose(1, 3).contiguous() / 255.0
        img = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                          std=(0.26862954, 0.24130258, 0.27577711))(img)
        clip_feats = self.clip_encode_image(img)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(img.shape[0], 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)
        
        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query


if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    
    device = 'cuda' # only support in gpu
    image = np.random.rand(1, 224, 224, 3)
    image = torch.from_numpy(image.astype(np.float16)).to(device)

    model = ImgModelWrapper().to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    model.eval()
    torch.onnx.export(model,
                      image,
                      'clip.onnx',
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}})
    
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import legacy
import numpy as np
import PIL.Image
import training.networks

def test():
    device = torch.device('npu')
    G = training.networks.Generator(z_dim=512, w_dim=512, c_dim=0, img_resolution=64, img_channels=3) # subclass of torch.nn.Module
    D = training.networks.Discriminator(c_dim=0, img_resolution=64, img_channels=3) # subclass of torch.nn.Module
    G = G.eval().requires_grad_(False).to(device)
    D = D.eval().requires_grad_(False).to(device)

    seed = 1
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    c = torch.zeros([1, G.c_dim], device=device)

    img = G(z, c)
    logits = D(img, c)

    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    img.show()
    print("Discriminator output:", logits)

if __name__ == '__main__':
    test()
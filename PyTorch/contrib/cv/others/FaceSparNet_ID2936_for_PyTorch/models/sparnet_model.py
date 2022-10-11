# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp
from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils
from models.sparnet import SPARNet

import apex
import os

CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != CALCULATE_DEVICE:
    CALCULATE_DEVICE = f'npu:{CALCULATE_DEVICE}'

class SPARNetModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        parser.add_argument('--scale_factor', type=int, default=8, help='upscale factor for sparnet')
        parser.add_argument('--lambda_pix', type=float, default=1.0, help='weight for pixel loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = SPARNet(res_depth=opt.res_depth, norm_type=opt.Gnorm, att_name=opt.att_name, bottleneck_size=opt.bottleneck_size) 
        self.netG = networks.define_network(opt, self.netG)
        self.model_names = ['G']
        self.load_model_names = ['G']
        self.loss_names = ['Pix'] 
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']

        if self.isTrain:
            self.criterionL1 = nn.L1Loss().to(CALCULATE_DEVICE)
            # self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizer_G = apex.optimizers.NpuFusedAdam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G]
            self.netG, self.optimizer_G = amp.initialize(self.netG, self.optimizer_G, opt_level='O2', loss_scale=128.0, combine_grad=True)

    def load_pretrain_model(self,):
        print('Loading pretrained model', self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.load_state_dict(weight)
    
    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(CALCULATE_DEVICE)
        self.img_HR = input['HR'].to(CALCULATE_DEVICE)

    def forward(self):
        # self.img_SR = self.netG(self.img_LR).to(self.opt.device)
        self.img_SR = self.netG(self.img_LR)

    def backward_G(self):
        # Pix loss
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        self.loss_Pix = self.loss_Pix.to(CALCULATE_DEVICE)
        #self.loss_Pix.backward()
        with amp.scale_loss(self.loss_Pix, self.optimizer_G) as scaled_loss:
            scaled_loss.backward()

    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
 
    def get_current_visuals(self, size=128):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]
        return visual_imgs
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
import apex
from torchvision import transforms
try:
	from apex import amp
except ImportError:
	amp = None


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):

	# os.environ['CUDA_VISIBLE_DEVICES']='0'

	device = torch.device(f'npu:{config.device_id}')
	torch.npu.set_device(device)
	print("Use NPU: {} for training".format(config.device_id))

	DCE_net = model.enhance_net_nopool().npu()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = apex.optimizers.NpuFusedAdam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	if config.apex:
		DCE_net, optimizer = amp.initialize(DCE_net, optimizer,
										opt_level=config.apex_opt_level,
										loss_scale=config.loss_scale_value,
										combine_grad=True)
	
	DCE_net.train()

	for epoch in range(config.num_epochs):
		cost_time = 0
		for iteration, img_lowlight in enumerate(train_loader):
			start_time = time.time()
			img_lowlight = img_lowlight.npu()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			#

			
			optimizer.zero_grad()

			if config.apex:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()
			#loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()
			cost_time += time.time() - start_time

			if ((iteration+1) % config.display_iter) == 0:
				time_average = cost_time / config.display_iter
				fps = config.train_batch_size / time_average
				#print("Loss at iteration", iteration+1, ":", loss.item())
				print("Iteration: {} Train_loss: {:.3f}, fps: {:3f}".format(iteration+1, loss.item(), fps))
				cost_time = 0
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		
		cost_time = 0



if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=128)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	# Mixed precision training parameters
	parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
	parser.add_argument('--apex-opt-level', default='O1', type=str, help='For apex mixed precision training'
																		 'O0 for FP32 training, O1 for mixed precision training.'
																		 'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
	parser.add_argument('--loss-scale-value', default=1024., type=float,
						help='loss scale using in amp, default -1 means dynamic')
	## for ascend 910
	parser.add_argument('--device_id', default=5, type=int, help='device id')

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	

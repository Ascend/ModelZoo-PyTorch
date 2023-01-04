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
# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import numpy as np 
from SETR.transformer_seg import SETRModel
from PIL import Image
import glob 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import apex
import torch.npu
#torch.npu.set_device('npu:7')
from datetime import datetime

from apex import amp
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='data_path')
	parser.add_argument('--data_path',default='./segmentation_car')
	parser.add_argument('--epoches',type=int,default=1)
	parser.add_argument('--device_id', default=5, type=int, help='device_id')
	parser.add_argument('--apex', action='store_true',
						help='User apex for mixed precision training')
	parser.add_argument('--apex-opt-level', default='O1', type=str,
						help='For apex mixed precision training'
							 'O0 for FP32 training, O1 for mixed precison training.')
	parser.add_argument('--loss-scale-value', default=1024., type=float,
						help='loss scale using in amp, default -1 means dynamic')
	args = parser.parse_args()
	return args

args = parse_args()

img_url = sorted(glob.glob( args.data_path + "/imgs/*"))
mask_url = sorted(glob.glob(args.data_path + "/masks/*"))
# print(img_url)
train_size = int(len(img_url) * 0.8)
train_img_url = img_url[:train_size]
train_mask_url = mask_url[:train_size]
val_img_url = img_url[train_size:]
val_mask_url = mask_url[train_size:]

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f'npu:{args.device_id}' if torch.npu.is_available() else "cpu")
torch.npu.set_device(device)
print("device is " + str(device))
out_channels = 1

def build_model():
	model = SETRModel(patch_size=(16, 16),
					in_channels=3,
					out_channels=1,
					hidden_size=1024,
					num_hidden_layers=6,
					num_attention_heads=16,
					decode_features=[512, 256, 128, 64])
	return model

class CarDataset(Dataset):
	def __init__(self, img_url, mask_url):
		super(CarDataset, self).__init__()
		self.img_url = img_url
		self.mask_url = mask_url

	def __getitem__(self, idx):
		img = Image.open(self.img_url[idx])
		img = img.resize((256, 256))
		img_array = np.array(img, dtype=np.float32) / 255
		mask = Image.open(self.mask_url[idx])
		mask = mask.resize((256, 256))
		mask = np.array(mask, dtype=np.float32)
		img_array = img_array.transpose(2, 0, 1)

		return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

	def __len__(self):
		return len(self.img_url)

def compute_dice(input, target):
	eps = 0.0001
	# input 鏄粡杩囦簡sigmoid 涔嬪悗鐨勮緭鍑恒
	input = (input > 0.5).float()
	target = (target > 0.5).float()

	# inter = torch.dot(input.view(-1), target.view(-1)) + eps
	inter = torch.sum(target.view(-1) * input.view(-1)) + eps

	# print(self.inter)
	union = torch.sum(input) + torch.sum(target) + eps

	t = (2 * inter.float()) / union.float()
	return t

def predict():
	model = build_model()
	model.load_state_dict(torch.load("./SETR/SETR_car.pkl", map_location="cpu"))
	print(model)

	import matplotlib.pyplot as plt
	val_dataset = CarDataset(val_img_url, val_mask_url)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	with torch.no_grad():
		for img, mask in val_loader:
			pred = torch.sigmoid(model(img))
			pred = (pred > 0.5).int()
			plt.subplot(1, 3, 1)
			print(img.shape)
			img = img.permute(0, 2, 3, 1)
			plt.imshow(img[0])
			plt.subplot(1, 3, 2)
			plt.imshow(pred[0].squeeze(0), cmap="gray")
			plt.subplot(1, 3, 3)
			plt.imshow(mask[0], cmap="gray")
			plt.show()

if __name__ == "__main__":

	model = build_model()
	#model.to(device)
	model.to(device)

	train_dataset = CarDataset(train_img_url, train_mask_url)
	train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=64)

	val_dataset = CarDataset(val_img_url, val_mask_url)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

	loss_func = nn.BCEWithLogitsLoss()
	#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
	#model, optimizer = amp.initialize(model, optimizer,opt_level="O1")
	optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=1e-5, weight_decay=1e-5)
	if args.apex:
		model, optimizer = amp.initialize(model, optimizer,
										  opt_level=args.apex_opt_level,
										  loss_scale=args.loss_scale_value,
										  combine_grad=True)
	step = 0
	report_loss = 0.0
	for epoch in range(args.epoches):
		print("epoch is " + str(epoch))
		a=datetime.now()
		for img, mask in tqdm(train_loader, total=len(train_loader)):
			start_time = datetime.now().timestamp()
			optimizer.zero_grad()
			step += 1
			img = img.to(device)
			mask = mask.to(device)

			pred_img = model(img) ## pred_img (batch, len, channel, W, H)
			if out_channels == 1:
				pred_img = pred_img.squeeze(1) # 鍘绘帀閫氶亾缁村害

			loss = loss_func(pred_img, mask)
			report_loss += loss.item()
			#loss.backward()
			if args.apex:
				with amp.scale_loss(loss,optimizer) as scale_loss:
					scale_loss.backward()
			else:
				loss.backward()
			optimizer.step()
			if step < 3:
				step_time = datetime.now().timestamp()-start_time
				print("step_time = {:.4f}".format(step_time), flush=True)
			if step % len(train_loader) == 0:
				b=datetime.now()
				print("FPS is " + str(len(train_loader)/(b-a).seconds))
				dice = 0.0
				n = 0
				model.eval()
				with torch.no_grad():
					print("report_loss is " + str(report_loss))
					report_loss = 0.0
					for val_img, val_mask in tqdm(val_loader, total=len(val_loader)):
						n += 1
						val_img = val_img.to(device)
						val_mask = val_mask.to(device)
						pred_img = torch.sigmoid(model(val_img))
						if out_channels == 1:
							pred_img = pred_img.squeeze(1)
						cur_dice = compute_dice(pred_img, val_mask)
						dice += cur_dice
					dice = dice / n
					print("mean dice is " + str(dice))
					torch.save(model.state_dict(), "./SETR/SETR_car.pkl")
					model.train()

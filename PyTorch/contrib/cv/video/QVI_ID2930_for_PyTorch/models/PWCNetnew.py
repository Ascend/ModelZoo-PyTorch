# Copyright 2020 Huawei Technologies Co., Ltd
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
import math

from . import correlation


class PWCNet(torch.nn.Module):
	def __init__(self):
		super(PWCNet, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()
				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, x):
				tensor_one = self.moduleOne(x)
				tensor_two = self.moduleTwo(tensor_one)
				tensor_thr = self.moduleThr(tensor_two)
				tensor_fou = self.moduleFou(tensor_thr)
				tensor_fiv = self.moduleFiv(tensor_fou)
				tensor_six = self.moduleSix(tensor_fiv)
				return [tensor_one, tensor_two, tensor_thr, tensor_fou, tensor_fiv, tensor_six]

		class Decoder(torch.nn.Module):
			def __init__(self, int_level):
				super(Decoder, self).__init__()
				int_previous = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][int_level + 1]
				int_current = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][int_level + 0]

				if int_level < 6:
					self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if int_level < 6:
					self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=int_previous + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if int_level < 6:
					self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][int_level + 1]

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=int_current, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=int_current + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=int_current + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=int_current + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=int_current + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
				self.moduleSix = torch.nn.Sequential(torch.nn.Conv2d(in_channels=int_current + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1))

			def backward(self, tensor_input, tensor_flow, backward_tensor_grid, backward_tensor_partial):
				if str(tensor_flow.size()) not in backward_tensor_grid:
					tensor_horizontal = torch.linspace(-1.0, 1.0, tensor_flow.size(3)).view(1, 1, 1, tensor_flow.size(3)).expand(tensor_flow.size(0), -1, tensor_flow.size(2), -1)
					tensor_vertical = torch.linspace(-1.0, 1.0, tensor_flow.size(2)).view(1, 1, tensor_flow.size(2), 1).expand(tensor_flow.size(0), -1, -1, tensor_flow.size(3))
					backward_tensor_grid[str(tensor_flow.size())] = torch.cat([tensor_horizontal, tensor_vertical], 1).npu()

				if str(tensor_flow.size()) not in backward_tensor_partial:
					backward_tensor_partial[str(tensor_flow.size())] = tensor_flow.new_ones([tensor_flow.size(0), 1, tensor_flow.size(2), tensor_flow.size(3)])

				tensor_flow = torch.cat([tensor_flow[:, 0:1, :, :] / ((tensor_input.size(3) - 1.0) / 2.0), tensor_flow[:, 1:2, :, :] / ((tensor_input.size(2) - 1.0) / 2.0)], 1)
				tensor_input = torch.cat([tensor_input, backward_tensor_partial[str(tensor_flow.size())]], 1)
				tensor_output = torch.nn.functional.grid_sample(input=tensor_input, grid=(backward_tensor_grid[str(tensor_flow.size())] + tensor_flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
				tensor_mask = tensor_output[:, -1:, :, :]
				tensor_mask[tensor_mask > 0.999] = 1.0
				tensor_mask[tensor_mask < 1.0] = 0.0
				return tensor_output[:, :-1, :, :] * tensor_mask

			def forward(self, tensor_first, tensor_second, object_previous, backward_tensor_grid, backward_tensor_partial):
				tensor_feat = None
				if object_previous is None:
					tensor_volume = torch.nn.functional.leaky_relu(input=correlation.function_correlation(tensor_first=tensor_first, tensor_second=tensor_second), negative_slope=0.1, inplace=False)
					tensor_feat = torch.cat([tensor_volume], 1)
				elif object_previous is not None:
					tensor_flow = self.moduleUpflow(object_previous['tensorFlow'])
					tensor_feat = self.moduleUpfeat(object_previous['tensorFeat'])
					# CUPY
					tensor_volume = torch.nn.functional.leaky_relu(
						input=correlation.function_correlation(tensor_first=tensor_first, tensor_second=self.backward(tensor_second, tensor_flow * self.dblBackward, backward_tensor_grid, backward_tensor_partial)), negative_slope=0.1, inplace=False)
					tensor_feat = torch.cat([tensor_volume, tensor_first, tensor_flow, tensor_feat], 1)

				tensor_feat = torch.cat([self.moduleOne(tensor_feat), tensor_feat], 1)
				tensor_feat = torch.cat([self.moduleTwo(tensor_feat), tensor_feat], 1)
				tensor_feat = torch.cat([self.moduleThr(tensor_feat), tensor_feat], 1)
				tensor_feat = torch.cat([self.moduleFou(tensor_feat), tensor_feat], 1)
				tensor_feat = torch.cat([self.moduleFiv(tensor_feat), tensor_feat], 1)

				tensor_flow = self.moduleSix(tensor_feat)

				return {
					'tensorFlow': tensor_flow,
					'tensorFeat': tensor_feat
				}

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)

			def forward(self, tensor_input):
				return self.moduleMain(tensor_input)

		self.moduleExtractor = Extractor()
		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)
		self.moduleRefiner = Refiner()

	def forward(self, tensor_first, tensor_second):
		int_width = tensor_first.size(3)
		int_height = tensor_first.size(2)

		tensor_preprocessed_first = tensor_first
		tensor_preprocessed_second = tensor_second

		int_preprocessed_width = int(math.floor(math.ceil(int_width / 64.0) * 64.0))
		int_preprocessed_height = int(math.floor(math.ceil(int_height / 64.0) * 64.0))

		tensor_preprocessed_first = torch.nn.functional.interpolate(input=tensor_preprocessed_first, size=(int_preprocessed_height, int_preprocessed_width), mode='bilinear', align_corners=False)
		tensor_preprocessed_second = torch.nn.functional.interpolate(input=tensor_preprocessed_second, size=(int_preprocessed_height, int_preprocessed_width), mode='bilinear', align_corners=False)
		tensor_flow = 20.0 * torch.nn.functional.interpolate(input=self.forward_pre(tensor_preprocessed_first, tensor_preprocessed_second), size=(int_height, int_width), mode='bilinear', align_corners=False)
		tensor_flow[:, 0, :, :] *= float(int_width) / float(int_preprocessed_width)
		tensor_flow[:, 1, :, :] *= float(int_height) / float(int_preprocessed_height)

		return tensor_flow

	def forward_pre(self, tensor_first, tensor_second):

		backward_tensor_grid = {}
		backward_tensor_partial = {}

		tensor_first = self.moduleExtractor(tensor_first)
		tensor_second = self.moduleExtractor(tensor_second)

		object_estimate = self.moduleSix(tensor_first[-1], tensor_second[-1], None, backward_tensor_grid, backward_tensor_partial)
		object_estimate = self.moduleFiv(tensor_first[-2], tensor_second[-2], object_estimate, backward_tensor_grid, backward_tensor_partial)
		object_estimate = self.moduleFou(tensor_first[-3], tensor_second[-3], object_estimate, backward_tensor_grid, backward_tensor_partial)
		object_estimate = self.moduleThr(tensor_first[-4], tensor_second[-4], object_estimate, backward_tensor_grid, backward_tensor_partial)
		object_estimate = self.moduleTwo(tensor_first[-5], tensor_second[-5], object_estimate, backward_tensor_grid, backward_tensor_partial)

		return object_estimate['tensorFlow'] + self.moduleRefiner(object_estimate['tensorFeat'])

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
# This file implements acceleration/velocity calculation

import torch.nn as nn


class AcFusionLayer(nn.Module):
	"""docstring for AcFusionLayer"""
	def __init__(self, ):
		super(AcFusionLayer, self).__init__()
	
	def forward(self, flo10, flo12, flo21, flo23, t=0.5):
		"""
			-- input: four flows
			-- output: center shift
		"""

		return 0.5 * ((t + t**2)*flo12 - (t - t**2)*flo10), 0.5 * (((1 - t) + (1 - t)**2)*flo21 - ((1 - t) - (1 - t)**2)*flo23)
		

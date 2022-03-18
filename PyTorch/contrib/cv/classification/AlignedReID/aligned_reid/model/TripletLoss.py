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
# limitations under the License.from __future__ import print_function

from torch import nn
from torch.autograd import Variable


class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss

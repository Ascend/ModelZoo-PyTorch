# Copyright 2021 Huawei Technologies Co., Ltd
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

from .collater import Collater
from .icdar_dataset import IC15Dataset, IC13Dataset
from .voc_dataset import VOCDataset
from .hrsc_dataset import HRSCDataset
from .dota_dataset import DOTADataset
from .ucas_aod_dataset import UCAS_AODDataset
from .nwpu_vhr_dataset import NWPUDataset
from .gaofen_dataset import GaoFenShipDataset


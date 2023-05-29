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
rm -rf kernel_meta/
source pt_set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export NPUID=0
export HCCL_WHITELIST_DISABLE=1

export RANK=0
python3 ./tools/train.py configs/gfocal/gfocal_r50_fpn_1x.py \
	--cfg-options \
	optimizer.lr=0.005 \
	--seed 0 \
	--gpu-ids 0 \
	--opt-level O1 &


#PORT=27511 ./tools/dist_train.sh configs/gfocal/gfocal_r50_fpn_1x.py 1 --cfg-options optimizer.lr=0.00125 --seed 0 --gpu-ids 0 --opt-level O1
#PORT=27511 ./tools/dist_test.sh configs/gfocal/gfocal_r50_fpn_1x.py work_dirs/gfocal_r50_fpn_1x/epoch_1.pth 1 --eval bbox
#nohup python3 tools/test.py configs/gfocal/gfocal_r50_fpn_1x.py work_dirs/gfocal_r50_fpn_1x/epoch_1.pth --eval bbox
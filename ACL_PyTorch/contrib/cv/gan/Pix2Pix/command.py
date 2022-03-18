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

# 转换为om模型
# export install_path=/usr/local/Ascend/ascend-toolkit/latest
# export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
# export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
# export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
# export ASCEND_OPP_PATH=${install_path}/opp
# export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

# atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs16 --input_format=NCHW --input_shape="inputs:16,3,256,256" --log=debug --soc_version=Ascend310
# atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs4 --input_format=NCHW --input_shape="inputs:4,3,256,256" --log=debug --soc_version=Ascend310
# atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs8 --input_format=NCHW --input_shape="inputs:8,3,256,256" --log=debug --soc_version=Ascend310

# 测试集预处理
# python preprocess.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained 

# 生成预处理数据，生成对应的info文件
# python gen_dataset_info.py bin ./datasets/facades/bin ./datasets/facades/netG_prep_bin.info 256 256

# 离线推理

# ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./checkpoints/facades_label2photo_pretrained/netG_om_bs16.om -input_text_path=./netG_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
# ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=./checkpoints/facades_label2photo_pretrained/netG_om_bs4.om -input_text_path=./netG_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
# ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=8 -om_path=./checkpoints/facades_label2photo_pretrained/netG_om_bs8.om -input_text_path=./netG_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False


# 精度统计
# python pix2pix_eval_acc.py 


# 性能测试
# trtexec --onnx=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --fp16 --shapes=inputs:1x3x256x256
# trtexec --onnx=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --fp16 --shapes=inputs:16x3x256x256
# trtexec --onnx=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --fp16 --shapes=inputs:4x3x256x256

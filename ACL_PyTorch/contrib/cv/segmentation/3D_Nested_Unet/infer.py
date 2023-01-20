# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import sys
import glob
import argparse

DATA_NUM = 27

def changebin(bin_path):
    filelist=glob.glob(f"{bin_path}/*.bin")
    for filename in filelist:
        filename1=filename.replace("0.bin", "1.bin")
        os.system(f"mv {filename} {filename1}")

def verify(environment):
    inputlist = os.listdir(f"{environment}/input/")
    outputlist = os.listdir(f"{environment}/output/")
    
    i = DATA_NUM
    
    for filename in inputlist:
        for j,outname in enumerate(outputlist):
            if outname.replace(".nii.gz", "_0000.nii.gz") == filename:
                i -= 1
                outputlist.pop(j)
                continue
    if 0 == i:
      print(" activate inference ...")
    else:
      print("data error!!!")
      exit(-1)


def infer(args):
    verify(args.environment)
    inputlist = os.listdir(f"{args.environment}/input/")
    for filename in inputlist:
        tmpname = filename.replace("_0000.nii.gz", ".nii.gz",)
        os.system(f"mv {args.environment}/output/{tmpname} {args.environment}/output/{filename}")
        os.system(f"rm {args.environment}/input_bins/* -rf")
        os.system(f"{args.interpreter} 3d_nested_unet_preprocess.py --file_path {args.environment}/input_bins/")
        os.system(f"rm {args.environment}/result/bs1 -rf")
        os.system("{} --model={} --input={}/input_bins/  --output={}/result/ --output_dirname=bs1 --outfmt=BIN  --batchsize=1 --device={}" \
                  .format(args.npu_interpreter, args.om_path, args.environment, args.environment, args.device))
        os.system(f"rm {args.environment}/result/bs1/*[1-4].bin")
        changebin(f"{args.environment}/result/bs1/")
        os.system(f"{args.interpreter} 3d_nested_unet_postprocess.py --file_path {args.environment}/result/bs1/")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="/home/3D_Nested_Unet/environment")
    parser.add_argument("--interpreter", default="python3")
    parser.add_argument("--npu_interpreter", default="python3 ais_infer.py")
    parser.add_argument("--om_path", default="./nnunetplusplus.om")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    
    infer(args)

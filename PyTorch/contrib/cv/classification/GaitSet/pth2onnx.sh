#!/usr/bin/env bash
currentDir=$(cd "$(dirname "$0")";pwd)/..
# echo $currentDir
source $currentDir'/test/npu_set_env.sh'
python3.7.5 -u pth2onnx.py
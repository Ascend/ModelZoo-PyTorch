#!/usr/bin/env bash
currentDir=$(cd "$(dirname "$0")";pwd)/..
echo $currentDir
source $currentDir'/test/npu_set_env.sh'
export datasetPath=$currentDir'/../CASIA-B-Pre'
nohup python -u test_main.py --iter=40000 --batch_size 64 --cache=True | tee $currentDir'/logTestNpu.txt' &

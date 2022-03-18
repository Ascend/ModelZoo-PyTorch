#!/bin/bash

# 导入环境变量
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# usage
if [ $# -ne 4 ]
then
    echo "usage: sh run_profiling.sh <benchmark_path> <om_path> <res_save_path> <device_id>"
    exit 1
fi

# 清空终端
clear
echo "[INFO] Now, run run_profiling.sh ...\n"

# 变量赋值
benchmark_path=$1
om_path=$2
res_save_path=$3
device_id=$4

# 参数校验
if [ ! -e $benchmark_path ]
then
    echo "'$benchmark_path' not exist!"
    exit 1
fi

if [ ! -e $om_path ]
then
    echo "'$om_path' not exist!"
    exit 1
fi

if [ $device_id -lt 0 ]
then
    echo "'device_id' must >= 0"
    exit 1
fi

# 删除然后创建结果保存目录
if [ -e $res_save_path ]
then
    rm -rf $res_save_path
    mkdir $res_save_path
else
    mkdir $res_save_path
fi

# 运行profiling工具
# + 推理
echo "[INFO] Step 1: Infering ..."
$install_path/x86_64-linux/tools/profiler/bin/msprof --output=$res_save_path \
    --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --sys-io-profiling=on \
    --dvpp-profiling=off --runtime-api=on --task-time=on --model-execution=on --aicpu=on \
    --application="$benchmark_path --round=10 -om_path=$om_path -device_id=$device_id -batch_size=1"
echo

# + 分析
echo "[INFO] Step 2: Analyzing ..."
python3.7 $install_path/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.py import \
    -dir $res_save_path
echo

# + export summary
echo "[INFO] Step 3: Export summary ..."
python3.7 $install_path/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.py export summary \
    -dir $res_save_path --iteration-id 1
echo

# + export timeline
echo "[INFO] Step 4: Export timeline ..."
python3.7 $install_path/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.py export timeline \
    -dir $res_save_path --iteration-id 1
echo

# + 删除多余目录
rm -rf $res_save_path/log $res_save_path/sqlite ./result

# 结束
echo "[INFO] Done!"
exit 0

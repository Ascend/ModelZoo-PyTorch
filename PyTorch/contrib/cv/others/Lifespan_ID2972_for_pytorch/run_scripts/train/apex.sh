#!/bin/bash

CUR_DIR=$( cd `dirname $0` ; pwd )
SCRIPT_DIR=$( cd `dirname $CUR_DIR` ; pwd )
ROOT_DIR=$( cd `dirname $SCRIPT_DIR` ; pwd )
APEX_DIR=$ROOT_DIR/apex




#if [ ! -d "./apex" ];then
#  echo "Git Clone Apex form $APEX_NPU_URL"
#  git clone $APEX_NPU_URL
#  git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
#
#  if [ ! '$?' = 0 ];then
#    echo "clone status : Failed"
#    exit 1
#  fi
#
#fi

cd $APEX_DIR

pip3 install -r requirements.txt
pip3 install -r requirements_dev.txt

if [ '$?' = 0 ];then
  exit 1
fi

echo "complie APEX."
python3 setup.py --cpp_ext --npu_float_status bdist_wheel
if [ '$?' = 0 ];then
  exit 1
fi
echo "reinstall APEX"
pip3 uninstall apex
#arch表示架构，为aarch64或x86_64
pip3 install --upgrade apex-0.1+ascend-cp37-cp37m-linux_aarch64.whl




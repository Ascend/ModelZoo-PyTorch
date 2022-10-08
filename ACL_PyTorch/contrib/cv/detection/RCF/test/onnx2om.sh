#!/bin/bash

rm -rf rcf_bs1_321x481.om
rm -rf rcf_bs1_481x321.om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=rcf_bs1_change_sim_321x481.onnx --output=rcf_bs1_321x481 \
--input_format=NCHW --input_shape="image:1,3,321,481" --log=debug --soc_version=Ascend310
atc --framework=5 --model=rcf_bs1_change_sim_481x321.onnx --output=rcf_bs1_481x321 \
--input_format=NCHW --input_shape="image:1,3,481,321" --log=debug --soc_version=Ascend310

if [ $? != 0 ]; then
    echo "onnx to om fail!"
else
    echo "onnx to om success!"
    exit -1
fi

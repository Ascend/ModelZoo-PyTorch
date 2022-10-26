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
#!/bin/bash

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
    
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
    
done

traverse_dir()
{
    local filepath=$1
    
    for file in `ls -a $filepath`
    do
        if [ -d ${filepath}/$file ]
        then
            if [ $file == 'bin' ]
            then
                check_suffix ${filepath}/$file
            elif [[ $file != '.' && $file != '..' ]]
            then
                #递归
                traverse_dir ${filepath}/$file
            fi
        fi
    done
}
 
 
##获取后缀为txt或ini的文件
check_suffix()
{
    file=$1
    echo $file
    temp=`echo  $file | tr -cd "[0-9]" `
    python3.7 ais_infer.py --model ./srgan_dynamic_bs$batch_size.om --input=$file --dymHW ${temp:3:3},${temp:0:3}  --batchsize=$batch_size --output=./result
}
 
rm -rf result
mkdir result
#测试指定目录  /data_output/ci/history
traverse_dir ./preprocess_data

python3.7 cal_per.py $batch_size
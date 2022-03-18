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
export PYTHONUNBUFFERED=1
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PAT

source path.sh
source cmd.sh

nlsyms=data/lang_1char/non_lang_syms.txt
dict=data/lang_1char/train_chars.txt

decode_dir=om_decode
result_label="${decode_dir}/data.json"
mkdir ${decode_dir}

python3.7 SpeechTransformer_eval.py --result_label $result_label > $decode_dir/decode.log

wait
local/score.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}

echo "====accuracy data===="
result=`grep -e Avg -e SPKR -m 2 ${decode_dir}/result.txt`
echo "${result}"
echo $result > acc.txt

echo "====performance data===="
FPS=`grep -a FPS ${decode_dir}/decode.log | awk '{print $NF}' | awk 'END {print}'` 
echo "FPS: ${FPS}"
echo $FPS > perf.txt

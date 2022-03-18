source ./test/env_npu.sh
. /usr/local/Ascend/ascend-toolkit/set_env.sh &&
./benchmark.x86_64 -model_type=bert -batch_size=1 -device_id=0 \
-om_path=./TinyBERT.om -input_text_path=./TinyBERT.info \
-output_binary=true
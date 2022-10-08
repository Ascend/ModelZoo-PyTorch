source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --model=$1 \
    --framework=5 \
    --output=$2 \
    --log=error \
    --soc_version=Ascend310

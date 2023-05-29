source ./test/env_npu.sh
export RANK_SIZE=8
RANK_ID_START=0

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    export RANK_ID=$RANK_ID
    echo $RANK_ID 
    nohup python3 -u train.py \
        --cfg ./intrada_8p.yml \
        --rank $RANK_ID \
        --device_type npu \
        --device_id $RANK_ID \
        --world_size $RANK_SIZE \
        --distributed \
        --performance_log &>performance_8p.log &
done
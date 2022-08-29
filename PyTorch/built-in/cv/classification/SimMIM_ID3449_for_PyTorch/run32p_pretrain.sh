source env_npu.sh
export WORLD_SIZE=32
export NUMS=8
# 集群中当前节点的节点序号
node_rank=0
#集群中主节点服务器ip
master_addr=x.x.x.x
# 服务器自身ip
export HCCL_IF_IP=x.x.x.x
HCCL_WHITELIST_DISABLE=1

rm -f nohup.out

for((RANK_ID=0;RANK_ID<NUMS;RANK_ID++));
do
    local_rank=$RANK_ID
    export RANK=$((RANK_ID + NUMS * node_rank))

    nohup python3 main_simmim.py  \
        --addr ${master_addr} \
        --cfg configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml \
        --batch-size 128 \
        --amp-opt-level O1 \
        --local_rank ${local_rank} \
        --data-path /data/imagenet/train &
done

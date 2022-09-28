source env_npu.sh
export WORLD_SIZE=1
rm -f nohup.out

RANK_ID=0

export RANK=$RANK_ID

nohup taskset -c 0-24 python3.7 main_simmim.py  \
    --cfg configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml \
    --opts TRAIN.EPOCHS 2 \
    --batch-size 128 \
    --amp-opt-level O1 \
    --local_rank $RANK_ID \
    --data-path /data/imagenet/train &

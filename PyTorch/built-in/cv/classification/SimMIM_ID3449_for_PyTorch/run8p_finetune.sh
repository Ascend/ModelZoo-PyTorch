source env_npu.sh
export WORLD_SIZE=8
rm -f nohup.out

KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<WORLD_SIZE;RANK_ID++));
do
    export RANK=$RANK_ID
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    nohup taskset -c $PID_START-$PID_END python3.7 main_finetune.py  \
        --cfg configs/swin_base__100ep/simmim_finetune__swin_base__img192_window6__100ep.yaml \
        --pretrained ./output/simmim_pretrain/simmim_pretrain__swin_base__img192_window6__100ep/ckpt_epoch_99.pth \
        --batch-size 128 \
        --amp-opt-level O1 \
        --local_rank $RANK_ID \
        --data-path /data/imagenet &
done

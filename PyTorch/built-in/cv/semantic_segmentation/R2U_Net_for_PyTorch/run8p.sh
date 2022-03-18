source env.sh
rm -rf ./models

export RANK_SIZE=8

for((RANK_ID=0;RANK_ID<8;RANK_ID++));
do
    export RANK_ID=$RANK_ID
    nohup python3 main.py  \
        --model_type R2U_Net \
        --data_path ./dataset \
        --batch_size 128 \
        --lr 0.0016 \
        --num_workers 128 \
        --apex 1 \
        --apex-opt-level O2 \
        --distributed \
        --loss_scale_value 1024 \
        --npu_idx $RANK_ID\
        --num_epochs 150 &
done
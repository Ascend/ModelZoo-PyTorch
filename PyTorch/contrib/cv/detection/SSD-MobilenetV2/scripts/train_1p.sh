# npu
source scripts/set_npu_env.sh
nohup \
python3 -u train_ssd.py \
  --dataset_type voc  \
  --data_path /opt/npu/voc \
  --net mb2-ssd-lite \
  --base_net models/mb2-imagenet-71_8.pth  \
  --scheduler cosine \
  --lr 0.01 \
  --batch_size 32 \
  --t_max 200 \
  --validation_epochs 5 \
  --num_epochs 10  \
  --checkpoint_folder models/1p  \
    --eval_dir  models/1p/eval \
  --amp \
  --device npu\
  --num_workers 16 \
  --gpu 0  > models/1p/log.txt 2>&1 &

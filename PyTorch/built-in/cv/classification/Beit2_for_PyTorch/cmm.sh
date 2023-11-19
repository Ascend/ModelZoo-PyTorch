#!/usr/bin/env bash


export INF_NAN_MODE_ENABLE=0
export OMP_NUM_THREADS=1

# change these path according to your environment information start
source  /home/xxx/ascend-toolkit/set_env.sh

data_path=/home/xxx/coco2017
output_dir=/home/xxxx/result
log_dir=/home/xxx/result
# Download the tokenizer weight from the following http link, and modify tokenizer_weight  to the file path
tokenizer_weight=/home/xxx/tokenizer_model/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth
#tokenizer_weight='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D'

# change these path according to your environment information end

model=beit_huge_5B_cls_pt
data_set=image_folder
model=beit_huge_5B_cls_pt
shared_lm_head=True
early_layers=6
head_layers=2
num_mask_patches=75
second_input_size=224
second_interpolation=bicubic
min_crop_scale=0.2
tokenizer_model=vqkd_encoder_base_decoder_3x768x12_clip
nnodes=2
nproc_per_node=8
batch_size=120
lr=3e-4
warmup_epochs=10
clip_grad=3.0
drop_path=0.1
layer_scale_init_value=1e-5
opt_betas='0.9 0.999'
opt_eps=1e-8
epochs=1600
save_ckpt_freq=20
codebook_size=8192


python -m torch.distributed.run  --nproc_per_node=$nproc_per_node  run_beitv2_pretraining.py \
        --codebook_size $codebook_size \
        --data_set $data_set \
        --data_path $data_path \
        --output_dir $output_dir  \
        --log_dir $log_dir  \
        --model $model  \
        --shared_lm_head $shared_lm_head  \
        --early_layers $early_layers  \
        --head_layers $head_layers  \
        --num_mask_patches $num_mask_patches  \
        --second_input_size $second_input_size  \
        --second_interpolation $second_interpolation  \
        --min_crop_scale $min_crop_scale  \
        --tokenizer_model $tokenizer_model  \
        --tokenizer_weight $tokenizer_weight  \
        --batch_size $batch_size   \
        --lr $lr  \
        --warmup_epochs $warmup_epochs  \
        --clip_grad $clip_grad  \
        --drop_path $drop_path  \
        --layer_scale_init_value $layer_scale_init_value  \
        --imagenet_default_mean_and_std  \
        --opt_betas $opt_betas  \
        --opt_eps $opt_eps  \
        --epochs $epochs  \
        --save_ckpt_freq $save_ckpt_freq

CHECKPOINT=adgen-chatglm-6b-ft-1e-4-test
STEP=5000
export TRAIN_STATE=0
source env_npu.sh
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ./output/$CHECKPOINT/checkpoint-$STEP  \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --fp16_full_eval \
    --NPU_VISIBLE_DEVICE 1

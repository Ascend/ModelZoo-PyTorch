
source env_npu.sh
python3 preprocess.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../model/ \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR \
    --max_source_length 1024 \
    --max_target_length 1024
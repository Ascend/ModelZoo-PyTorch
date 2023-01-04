source env.sh
export PYTHONPATH=${pwd}../../../src:$PYTHONPATH
python3 run_mlm.py \
        --model_type bert \
        --config_name bert-base-chinese/config.json \
        --tokenizer_name bert-base-chinese \
        --train_file ./train_huawei.txt \
        --line_by_line \
        --pad_to_max_length \
        --save_steps 5000 \
        --overwrite_output_dir \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --do_train \
        --fp16 \
        --fp16_opt_level O2 \
        --loss_scale 8192 \
        --use_combine_grad \
        --optim adamw_apex_fused_npu \
        --output_dir ./output
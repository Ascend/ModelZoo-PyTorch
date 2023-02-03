source env_npu.sh

python3 -m torch.distributed.launch --nproc_per_node=8 examples/run_xfun_re.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 2500 \
        --per_device_train_batch_size 2 \
        --warmup_ratio 0.1 \
        --fp16 \
        --fp16_backend apex
source ./env_npu.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export HCCL_WHITELIST_DISABLE=1
/usr/local/Ascend/driver/tools/msnpureport -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3 main.py \
    --video_path /data/resnet3d/hmdb51_jpg \
    --annotation_path /data/resnet3d/hmdb51_json/hmdb51_1.json \
    --result_path outputs \
    --dataset hmdb51 \
    --n_classes 51 \
    --n_pretrain_classes 700 \
    --pretrain_path r3d18_K_200ep.pth \
    --ft_begin_module fc \
    --model resnet \
    --model_depth 18 \
    --batch_size 1024 \
    --n_threads 128 \
    --checkpoint 5 \
    --amp_cfg \
    --opt_level O2 \
    --loss_scale_value 1024 \
    --distributed \
    --ngpus_per_node 8 \
    --device_list '0,1,2,3,4,5,6,7' \
    --manual_seed 1234 \
    --learning_rate 0.08 \
    --tensorboard &

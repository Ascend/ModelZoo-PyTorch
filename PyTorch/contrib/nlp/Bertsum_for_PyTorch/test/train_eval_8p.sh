source ../test/env_npu.sh
python train.py \
        -mode validate \
        -bert_data_path ../bert_data/cnndm \
        -model_path ../models/bert_classifier \
        -visible_gpus 0 \
        -batch_size 30000 \
        -log_file ../logs/bert_classifier \
        -result_path ../results \
        -test_all \
        -block_trigram true \
        -train_npu true \
        -train_with_amp > log.valid.txt 2>&1 &

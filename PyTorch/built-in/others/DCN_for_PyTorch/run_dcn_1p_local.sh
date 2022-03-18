source ./test/env.sh

python3.7 -u run_classification_criteo_dcn.py \
--npu_id=0 \
--trainval_path='path/to/criteo_trainval.txt' \
--test_path='path/to/criteo_test.txt' \
--lr=0.0001 \
--use_fp16

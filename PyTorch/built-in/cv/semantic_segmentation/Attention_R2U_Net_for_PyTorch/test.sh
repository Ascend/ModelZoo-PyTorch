source ./npu_env.sh
a='R2AttU_Net'
python3 main_1p.py --model_type=$a --data_path="./dataset/train" --mode='test' --test_model_path="path/to/model"
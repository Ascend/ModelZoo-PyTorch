source ./npu_env.sh
a='R2AttU_Net'
nohup python3 main_8p.py --model_type=$a --data_path="./dataset" > train_8p.log 2>&1 &
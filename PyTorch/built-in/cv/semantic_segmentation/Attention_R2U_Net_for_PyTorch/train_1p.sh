source ./npu_env.sh
a='R2AttU_Net'
nohup python3 main_1p.py --model_type=$a  --data_path="./dataset" > train_1p.log 2>&1 &
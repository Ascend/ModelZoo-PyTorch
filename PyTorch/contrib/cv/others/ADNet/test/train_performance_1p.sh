rm -rf train_performance_1p.log
nohup python3.7 -u train.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --DeviceID 0 --epochs 6 --is_distributed 0 --loss_scale 8 | tee -a train_performance_1p.log

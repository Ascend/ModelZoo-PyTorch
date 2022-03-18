export PYTHONPATH=$PYTHONPATH:./core
python3 -u train_cpu.py --name raft-kitti  --stage kitti  --gpus 0 1 --num_steps 2000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
#python3 -u train.py --name raft-kitti  --stage kitti --validation kitti --gpus 0 1 --num_steps 20 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
#python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001

export WORLD_SIZE=1
taskset -c 0-25 python3.7 train.py train.yaml 10

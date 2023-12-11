#!/bin/bash  
  
source /home/cann1115/ascend-toolkit/set_env.sh
   
python /home/FastChat-master/tasks/task_eval.py   \
       --model_path /home/FastChat-master/7B-vicuna/  \
       --test_dir /home/FastChat-master/tasks/Mmlu/  \
       --task Mmlu
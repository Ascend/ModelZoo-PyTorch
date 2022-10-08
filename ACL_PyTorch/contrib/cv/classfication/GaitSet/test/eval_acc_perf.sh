# post-process
#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.7.5 -u test.py --iter=-1 --batch_size 1 --cache=True --post_process=True

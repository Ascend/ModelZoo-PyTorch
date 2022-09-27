source env_npu.sh

python3 -m yolox.tools.train -n yolox-s -d 1 -b 16 --fp16 -f exps/example/yolox_voc/yolox_voc_s.py
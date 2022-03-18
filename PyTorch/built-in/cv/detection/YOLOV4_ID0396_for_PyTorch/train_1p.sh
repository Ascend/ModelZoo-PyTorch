source ./test/env.sh
taskset -c 0-23 python3.7 train.py --device_id 0 \
                                   --img 608 608 \
                                   --epochs=300 \
                                   --data coco.yaml \
                                   --cfg cfg/yolov4.cfg \
                                   --weights '' \
                                   --name yolov4 \
                                   --batch-size 32 \
                                   --amp \
                                   --opt-level O1 \
                                   --loss_scale 128 \
                                   --notest


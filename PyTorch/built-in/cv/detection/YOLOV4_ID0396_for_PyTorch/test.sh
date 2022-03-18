source ./test/env.sh
taskset -c 0-23  python3.7 test.py --img-size 608 --conf 0.001 --batch 32 --device 0 --data coco.yaml \
                                  --cfg cfg/yolov4.cfg --weights weights/last.pt

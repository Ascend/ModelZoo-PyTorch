cd mmdetection
mkdir data
ln -s /root/datasets/coco data/coco
python3 tools/test.py configs/ssd/ssd300_coco.py ../ssd300_coco_20200307-a92d2092.pth --eval bbox

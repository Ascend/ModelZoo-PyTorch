pip install mmdet==2.8.0
pip install mmcv-full==1.2.4
pip install mmpycocotools==12.0.3
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
pip install -r requirements/build.txt
pip install -v -e .
patch -p1 < ../ssd_mmdetection.diff
cd ..
echo "install mmdetection successfully"
wget http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth
echo "download SSD300 pth successfully"
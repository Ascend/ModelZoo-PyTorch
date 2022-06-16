python mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py mmdetection/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth   --output-file faster_rcnn_r50_fpn_1x_coco.onnx  --shape 1216 --show &&
python  modifyonnx.py --batch_size=1 &&
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
atc --framework=5 --model=./faster_rcnn_r50_fpn_1x_coco_change_bs1.onnx --output=./faster_rcnn_r50_fpn_1x_coco_bs1  --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=debug --soc_version=Ascend${chip_name}
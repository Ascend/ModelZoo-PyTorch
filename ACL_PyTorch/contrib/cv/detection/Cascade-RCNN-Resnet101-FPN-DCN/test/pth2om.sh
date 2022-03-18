python mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py ./cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth --output-file=cascadeR101dcn.onnx --shape=1216

bash atc.sh cascadeR101dcn.onnx cascadeR101dcn

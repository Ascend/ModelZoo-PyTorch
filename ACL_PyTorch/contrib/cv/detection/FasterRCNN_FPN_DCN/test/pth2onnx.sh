export PYTHONPATH=/root/anaconda3/envs/env_wgzheng/bin/python


python3.7 mmdetection/tools/pytorch2onnx.py \
          mmdetection/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
          mmdetection/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth   \
	  --output-file faster_rcnn_r50_fpn_1x_coco.onnx  \
          --shape 1216 \
          --show
          

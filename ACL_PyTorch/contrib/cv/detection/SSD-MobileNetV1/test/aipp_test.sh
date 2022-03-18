#! /bin/bash

atc --framework=5 --model=mb1-ssd_fix.onnx --input_format=NCHW --input_shape="image:1,3,304,304" --output=mb1-ssd_fix_aipp_bs1 --log=debug --soc_version=Ascend310 --insert_op_conf=aipp_ssd_pth.config


python3.7 get_info.py jpg /root/datasets/VOC2007/JPEGImages mb-ssd_prep_jpg.info


./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=1 -om_path=mb1-ssd_fix_aipp_bs1.om -input_text_path=mb-ssd_prep_jpg.info -input_width=304 -input_height=304 -output_binary=True -useDvpp=True


python3.7 SSD_MobileNet_postprocess.py /root/datasets/VOC2007 voc-model-labels.txt ./result/dumpOutput_device3/ ./eval_results3/
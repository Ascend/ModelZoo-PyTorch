export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
chmod u+x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -om_path=faster_rcnn_r50_fpn_1x_coco_bs1.om -device_id=0 -batch_size=1 -input_text_path=coco2017_bin.info -input_width=1216 -input_height=1216 -useDvpp=false -output_binary=true

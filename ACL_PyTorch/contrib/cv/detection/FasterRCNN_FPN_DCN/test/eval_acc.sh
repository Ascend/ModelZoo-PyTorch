python3.7 FasterRCNN+FPN+DCN_preprocess.py --image_folder_path ./data/coco/val2017 --bin_folder_path coco2017_bin &&
python3.7 gen_dataset_info.py bin coco2017_bin coco2017_bin.info 1216 1216  &&
python3.7 gen_dataset_info.py jpg ./data/coco/val2017 coco2017_jpg.info &&
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
chmod u+x benchmark.x86_64 &&
./benchmark.x86_64  -model_type=vision  -batch_size=1  -device_id=0 -input_text_path=coco2017_bin.info  -input_width=1216  -input_height=1216  -om_path=./faster_rcnn_r50_fpn_1x_coco_bs1.om -useDvpp=false --output_binary=true &&
python3.7 FasterRCNN+FPN+DCN_postprocess.py --test_annotation coco2017_jpg.info --bin_data_path result/dumpOutput_device0 &&
python3.7 txt2json.py --npu_txt_path detection-results --json_output_file coco_detection_result &&
python3.7 coco_eval.py --ground_truth data/coco/annotations/instances_val2017.json --detection_result coco_detection_result.json

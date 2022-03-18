./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017.info -input_width=1216 -input_height=1216 -useDvpp=False -output_binary=true -om_path=cascadeR101dcn.om

python mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info

python txt_to_json.py

python coco_eval.py

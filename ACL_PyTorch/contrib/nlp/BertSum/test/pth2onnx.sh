cd ../
python BertSum-pth2onnx.py -mode test -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH -visible_gpus -1 -gpu_ranks 0 -batch_size 1 -log_file LOG_FILE -result_path RESULT_PATH -test_all -block_trigram true -onnx_path bertsum_13000_9_bs1.onnx -path model_step_13000.pt


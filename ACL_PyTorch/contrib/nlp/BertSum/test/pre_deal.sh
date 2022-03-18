cd ../
python BertSum_pth_preprocess.py -mode test -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH -visible_gpus -1 -gpu_ranks 0 -batch_size 600 -log_file LOG_FILE -result_path RESULT_PATH -test_all -block_trigram true

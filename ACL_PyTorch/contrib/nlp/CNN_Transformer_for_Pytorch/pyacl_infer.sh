source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3.7 CNN_Transformer_pyacl_infer.py --model_path=./models/wav2vec2-base-960h.om --device_id=0 --cpu_run=True --sync_infer=True --workspace=0 --input_info_file_path=./pre_data/validation/bin_file.info --input_dtypes=float32 --infer_res_save_path=./om_infer_res_clean --res_save_type=bin

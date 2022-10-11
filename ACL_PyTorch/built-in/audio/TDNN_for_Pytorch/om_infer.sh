export PYTHONUNBUFFERD=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bs=$1

python3.7 tdnn_pyacl_infer.py --model_path=tdnn_bs${bs}s.om --batch_size=${bs} --device_id=0 --cpu_run=True --sync_infer=True --workspace=10 --input_info_file_path=mini_librispeech_test.info --input_dtypes=float32 --infer_res_save_path=./result --res_save_type=bin
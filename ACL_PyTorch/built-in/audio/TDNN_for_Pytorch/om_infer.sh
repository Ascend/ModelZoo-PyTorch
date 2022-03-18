install_path=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONUNBUFFERD=1
export PYTHONPATH=${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/acllib/lib64/:$LD_LIBRARY_PATH

python3.7 pyacl_infer.py --model_path=./tdnn_dynamic.om --device_id=4 --cpu_run=True --sync_infer=True --workspace=10 --input_info_file_path=mini_librispeech_test.input_info_file_path --input_dtypes=float32 --infer_res_save_path=./result --res_save_type=bin
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:${LD_LIBRARY_PATH}	
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/5.0.2/x86_64-linux/acllib/lib64/stub/	
export ASCEND_OPP_PATH=${install_path}/opp
alias python='/usr/local/Python-3.7.3/python'
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
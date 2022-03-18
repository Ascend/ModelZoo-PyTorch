source /home/package/5.0.2.alpha005/ascend-toolkit/set_env.sh 
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:${LD_LIBRARY_PATH}
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH

export ASCEN_SLOG_PRINT_TO_STDOUT=1
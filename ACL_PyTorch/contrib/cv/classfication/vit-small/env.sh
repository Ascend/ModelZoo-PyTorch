export install_path=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/toolkit/python/site-packages:${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:${install_path}/atc/bin:${install_path}/atc/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=${install_path}
export ASCEND_OPP_PATH=${install_path}/opp
export TOOLCHAIN_HOME=${install_path}/toolkit
export REPEAT_TUNE=True
export ENABLE_TUNE_BANK=True


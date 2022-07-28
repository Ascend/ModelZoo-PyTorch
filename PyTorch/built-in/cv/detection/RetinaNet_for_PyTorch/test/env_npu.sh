if [ -d /usr/local/Ascend/nnae/latest ];then
	export LD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH
    export PATH=$PATH:/usr/local/Ascend/nnae/latest/fwkacllib/ccec_compiler/bin/:/usr/local/Ascend/nnae/latest/toolkit/tools/ide_daemon/bin/
    export ASCEND_OPP_PATH=/usr/local/Ascend/nnae/latest/opp/
    export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
    export PYTHONPATH=/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
    export ASCEND_AICPU_PATH=/usr/local/Ascend/nnae/latest
else
	export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
	export PATH=$PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/ide_daemon/bin/
	export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/
	export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
	export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
    export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

export install_path=/usr/local/Ascend
#设置device侧日志登记为error
${install_path}/driver/tools/msnpureport -g error -d 0
${install_path}/driver/tools/msnpureport -g error -d 1
${install_path}/driver/tools/msnpureport -g error -d 2
${install_path}/driver/tools/msnpureport -g error -d 3
${install_path}/driver/tools/msnpureport -g error -d 4
${install_path}/driver/tools/msnpureport -g error -d 5
${install_path}/driver/tools/msnpureport -g error -d 6
${install_path}/driver/tools/msnpureport -g error -d 7
#关闭Device侧Event日志
${install_path}/driver/tools/msnpureport -e disable

path_lib=$(python3.7 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('-packages', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'

        result+=sys.path[index] + '/torch/lib:'
print(result)"""
)

echo ${path_lib}

export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH

#echo ${LD_LIBRARY_PATH}

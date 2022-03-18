export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH} 

./main test.jpg

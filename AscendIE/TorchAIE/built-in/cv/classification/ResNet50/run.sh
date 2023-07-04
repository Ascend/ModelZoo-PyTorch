rm -rf build
mkdir build
cd build
# 请根据实际运行环境修改DTORCH和DTORCH_AIE的路径
cmake .. -DTORCH=/usr/lib/python3.9.0/lib/python3.9/site-packages/torch -DTORCH_AIE=/usr/lib/python3.9.0/lib/python3.9/site-packages/torch_aie
make -j 32
./sample "../resnet50.ts"
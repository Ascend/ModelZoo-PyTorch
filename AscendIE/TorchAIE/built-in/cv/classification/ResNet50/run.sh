rm -rf build
mkdir build
cd build
cmake .. -DTORCH=/usr/lib/python3.9.0/lib/python3.9/site-packages/torch -DTORCH_AIE=/usr/lib/python3.9.0/lib/python3.9/site-packages/torch_aie
make -j 32
./sample "../resnet50.ts"
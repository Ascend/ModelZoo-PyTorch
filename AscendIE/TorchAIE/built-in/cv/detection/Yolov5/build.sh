rm -rf build
cmake -DCMAKE_PREFIX_PATH=${libtorchInstallPath} -B build .
cmake --build build -j`nproc`
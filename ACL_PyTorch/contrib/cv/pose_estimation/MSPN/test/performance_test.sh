#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./benchmark.x86_64 -round=20 -om_path=MSPN_bs16.om -device_id=0 -batch_size=16
./benchmark.x86_64 -round=20 -om_path=MSPN_bs16.om -device_id=0 -batch_size=1
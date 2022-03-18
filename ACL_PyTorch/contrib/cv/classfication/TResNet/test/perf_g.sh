#! /bin/bash
./benchmark.x86_64 -round=20 -om_path=tresnet_patch16_224_bs1.om  -device_id=0 -batch_size=1
./benchmark.x86_64 -round=20 -om_path=tresnet_patch16_224_bs16.om  -device_id=0 -batch_size=16
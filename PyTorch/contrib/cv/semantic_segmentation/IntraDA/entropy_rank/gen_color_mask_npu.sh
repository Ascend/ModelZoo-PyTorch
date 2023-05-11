#!/bin/bash
source ../intrada/test/env_npu.sh
python3 entropy.py --device_type npu --device_id 0 --checkpoint ../ADVENT/pretrained_models/gta2cityscapes_advent.pth

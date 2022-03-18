#!/bin/bash

source env_npu.sh

python3 proc_node_module.py

nohup python3 -m demo.py &
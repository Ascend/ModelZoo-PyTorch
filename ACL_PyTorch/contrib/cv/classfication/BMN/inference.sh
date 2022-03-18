#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)
echo $currentDir
python3.7 8p_1p_inference.py --mode inference | tee 8p_1p_inference.log &
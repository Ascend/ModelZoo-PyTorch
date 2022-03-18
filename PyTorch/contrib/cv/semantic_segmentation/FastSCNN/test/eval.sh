#!/usr/bin/env bash
python3.7 -u ./tools/eval.py --config-file configs/cityscapes_fast_scnn.yaml TEST.TEST_MODEL_PATH runs/checkpoints/FastSCNN__cityscape_2021-08-18-12-07/best_model.pth

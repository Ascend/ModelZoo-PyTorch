#!/usr/bin/env bash
python3 -u ./tools/eval.py --config-file configs/cityscapes_fast_scnn.yaml TEST.TEST_MODEL_PATH runs/checkpoints/FastSCNN__cityscape/best_model.pth

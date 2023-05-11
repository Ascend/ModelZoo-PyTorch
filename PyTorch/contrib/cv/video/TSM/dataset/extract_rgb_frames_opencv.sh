#!/usr/bin/env bash

python3 build_rawframes.py ./ucf101/videos/ ./ucf101/rawframes/ --task rgb --level 2 --ext avi --use-opencv
echo "Genearte raw frames (RGB only)"

cd ucf101/

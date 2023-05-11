#!/usr/bin/env bash

cd ../

PYTHONPATH=. python3 dataset/build_file_list.py ucf101 dataset/ucf101/videos/ --level 2 --format videos --shuffle
echo "Filelist for videos generated."

cd dataset/ucf101/

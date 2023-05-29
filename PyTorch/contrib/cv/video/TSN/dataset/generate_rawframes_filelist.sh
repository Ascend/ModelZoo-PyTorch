#!/usr/bin/env bash

cd ../

PYTHONPATH=. python3 dataset/build_file_list.py ucf101 dataset/ucf101/rawframes/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd dataset/ucf101/

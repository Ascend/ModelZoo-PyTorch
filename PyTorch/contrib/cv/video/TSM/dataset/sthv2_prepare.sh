source test/env_npu.sh

python3 dataset/build_rawframes.py data/sthv2/videos/ data/sthv2/rawframes/ --task rgb --level 1 --ext webm --use-opencv

python3 dataset/build_file_list.py sthv2 data/sthv2/rawframes/ --num-split 1 --level 1 --subset train --format rawframes --shuffle
python3 dataset/build_file_list.py sthv2 data/sthv2/rawframes/ --num-split 1 --level 1 --subset val --format rawframes --shuffle

python3 dataset/build_file_list.py sthv2 data/sthv2/videos --num-split 1 --level 1 --subset train --format videos --shuffle
python3 dataset/build_file_list.py sthv2 data/sthv2/videos --num-split 1 --level 1 --subset val --format videos --shuffle
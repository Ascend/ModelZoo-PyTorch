#!/bin/bash
set -eu
set -x

datasets_path="/home/data/modelnet40_normal_resampled"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# env
source env.sh
chmod a+x msame

# bs1 part1 preprocess
python pointnetplus_preprocess.py --preprocess_part 1 --save_path ./modelnet40_processed/bs1/pointset_chg_part1 --save_path2 ./modelnet40_processed/bs1/xyz_chg_part1 --data_loc $datasets_path

# bs1 part1 infer
output_dir='./out/bs1/part1'
mkdir -p $output_dir
./msame --model Pointnetplus_part1_bs1.om --input './modelnet40_processed/bs1/xyz_chg_part1,./modelnet40_processed/bs1/pointset_chg_part1' --output $output_dir --outfmt BIN

# bs1 part2 preprocess
python pointnetplus_preprocess.py --preprocess_part 2 --save_path ./modelnet40_processed/bs1/pointset_chg_part2 --save_path2 ./modelnet40_processed/bs1/xyz_chg_part2 --data_loc $output_dir --data_loc2 ./modelnet40_processed/bs1/xyz_chg_part1

# bs1 part2 infer
output_dir='./out/bs1/part2'
mkdir -p $output_dir
./msame --model Pointnetplus_part2_bs1.om --input './modelnet40_processed/bs1/xyz_chg_part2,./modelnet40_processed/bs1/pointset_chg_part2' --output $output_dir --outfmt BIN

# bs1 postprocess
python pointnetplus_postprocess.py --target_path $output_dir --data_loc $datasets_path

# bs16 part1 preprocess
python pointnetplus_preprocess.py --preprocess_part 1 --save_path ./modelnet40_processed/bs16/pointset_chg_part1 --save_path2 ./modelnet40_processed/bs16/xyz_chg_part1 --data_loc $datasets_path --batch_size 16

# bs16 part1 infer
output_dir='./out/bs16/part1'
mkdir -p $output_dir
./msame --model Pointnetplus_part1_bs16.om --input './modelnet40_processed/bs16/xyz_chg_part1,./modelnet40_processed/bs16/pointset_chg_part1' --output $output_dir --outfmt BIN

# bs16 part2 preprocess
python pointnetplus_preprocess.py --preprocess_part 2 --save_path ./modelnet40_processed/bs16/pointset_chg_part2 --save_path2 ./modelnet40_processed/bs16/xyz_chg_part2 --data_loc $output_dir --data_loc2 ./modelnet40_processed/bs16/xyz_chg_part1 --batch_size 16

# bs16 part2 infer
output_dir='./out/bs16/part2'
mkdir -p $output_dir
./msame --model Pointnetplus_part2_bs16.om --input './modelnet40_processed/bs16/xyz_chg_part2,./modelnet40_processed/bs16/pointset_chg_part2' --output $output_dir --outfmt BIN

# bs16 postprocess
python pointnetplus_postprocess.py --target_path $output_dir --data_loc $datasets_path --batch_size 16

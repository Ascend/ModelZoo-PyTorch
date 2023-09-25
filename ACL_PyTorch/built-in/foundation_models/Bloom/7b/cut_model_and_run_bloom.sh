#! /bin/bash
export HCCL_BUFFSIZE=110
input_dir="./model"
output_dir="./model_cut"
world_size_=2
MODELING_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/bloom
UTILS_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/generation
CODE_PATH=`pwd`
cp $UTILS_PATH/utils.py $UTILS_PATH/utils.py.bak
cp $MODELING_PATH/modeling_bloom.py $MODELING_PATH/modeling_bloom.py.bak
cp .patches/utils.patch $UTILS_PATH/
cd $UTILS_PATH
patch < -p1 utils.patch
cd $CODE_PATH

if [ ! -d "$output_dir" ];
then
    cp patches/parallel.patch $MODELING_PATH/
    cd $MODELING_PATH
    patch < -p1 parallel.patch
    cd $CODE_PATH
    echo "Cutted Weight directory does not exist, cuting the weight......"
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_
fi 
cp patches/performance.patch $MODELING_PATH/
cd $MODELING_PATH
patch < -p1 performance.patch
cd $CODE_PATH
echo "Weight directory exists, runing......"
torchrun --nproc_per_node 2 run_bloom_half_parallel_loadPartModel.py --load_path $output_dir

cp $UTILS_PATH/utils.py.bak $UTILS_PATH/utils.py
cp $MODELING_PATH/modeling_bloom.py.bak $MODELING_PATH/modeling_bloom.py
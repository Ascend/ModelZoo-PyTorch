ais_infer=$1
model_320=$2
model_620=$3
image_bin=$4
results=$5
batch_size=$6

rm -rf ${results}
mkdir ${results}

rm -rf temp
mkdir temp

python3 -m ais_bench --model=${model_320} --input=${image_bin}/48_320/ --dymBatch=${batch_size} --output=./temp 

mv ./temp/*/*.bin ${results}/
rm -rf ./temp/*

python3 -m ais_bench --model=${model_620} --input=${image_bin}/48_620/ --dymBatch=${batch_size} --output=./temp

mv ./temp/*/*.bin ${results}/
rm -rf ./temp

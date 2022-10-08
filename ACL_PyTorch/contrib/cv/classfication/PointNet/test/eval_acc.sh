source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "batch_size=1时:"
python3.7 pointnet_preprocess.py data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file batch_size=1
./tools/msame/out/msame --model "pointnet_bs1_fixed.om" --input "bin_file" --output "res_data/bs1_out/" --outfmt TXT --device 1
python3.7 pointnet_postprocess.py ./name2label.txt ./res_data/bs1_out batch_size=1

echo "batch_size=16时:"
python3.7 pointnet_preprocess.py data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file_bs16 batch_size=16
./tools/msame/out/msame --model "pointnet_bs16_fixed.om" --input "bin_file_bs16" --output "res_data/bs16_out/" --outfmt TXT --device 1
python3.7 pointnet_postprocess.py ./name2label.txt ./res_data/bs16_out batch_size=16

echo "pth模型评测中..."
python3.7 eval.py

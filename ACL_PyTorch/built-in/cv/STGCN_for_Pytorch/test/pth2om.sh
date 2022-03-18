source env.sh

rm -rf st_gcn_bs1.onnx
Python3.7 st_gcn_export.py –ckpt=./checkpoints/st_gcn.kinetics-6fa43f73.pth –onnx=./st_gcn_bs1.onnx –batch_size=1
Python3.7 st_gcn_export.py –ckpt=./checkpoints/st_gcn.kinetics-6fa43f73.pth –onnx=./st_gcn_bs16.onnx –batch_size=16
if [ -f "st_gcn_bs1.onnx" ] && [ -f "st_gcn_bs16.onnx" ]; then
  echo "onnx export success"
else
  echo "onnx export failed"
  exit -1
fi

rm -rf dcgan_sim_bs1.om dcgan_sim_bs16.om
atc --model=st_gcn_bs1.onnx --framework=5 --output=st_gcn_bs1 --input_format=ND --input_shape="actual_input_1:1,3,300,18,2" --soc_version=Ascend310
atc --model=st_gcn_bs16.onnx --framework=5 --output=st_gcn_bs16 --input_format=ND --input_shape="actual_input_1:16,3,300,18,2" --soc_version=Ascend310
if [ -f "st_gcn_bs1.om" ] && [ -f "st_gcn_bs16.om" ]; then
  echo "om export success"
else
  echo "om export failed"
  exit -1
fi
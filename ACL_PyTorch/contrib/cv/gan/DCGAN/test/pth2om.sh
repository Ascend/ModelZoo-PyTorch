source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf dcgan.onnx
python3.7 dcgan_pth2onnx.py checkpoint-amp-epoch_200.pth dcgan.onnx

if [ -f "dcgan.onnx" ]; then
  echo "onnx export success"
else
  echo "onnx export failed"
  exit -1
fi

rm -rf dcgan_sim_bs1.onnx dcgan_sim_bs16.onnx
python3.7 -m onnxsim --input-shape=1,100,1,1 dcgan.onnx dcgan_sim_bs1.onnx
python3.7 -m onnxsim --input-shape=16,100,1,1 dcgan.onnx dcgan_sim_bs16.onnx

if [ -f "dcgan_sim_bs1.onnx" ] && [ -f "dcgan_sim_bs16.onnx" ]; then
  echo "sim_onnx export success"
else
  echo "sim_onnx export failed"
  exit -1
fi

rm -rf dcgan_sim_bs1.om dcgan_sim_bs16.om
atc --framework=5 --model=./dcgan_sim_bs1.onnx --output=dcgan_sim_bs1 --input_format=NCHW --input_shape="noise:1,100,1,1" --log=debug --soc_version=Ascend310
atc --framework=5 --model=./dcgan_sim_bs16.onnx --output=dcgan_sim_bs16 --input_format=NCHW --input_shape="noise:16,100,1,1" --log=debug --soc_version=Ascend310

if [ -f "dcgan_sim_bs1.om" ] && [ -f "dcgan_sim_bs16.om" ]; then
  echo "sim_om export success"
else
  echo "sim_om export failed"
  exit -1
fi
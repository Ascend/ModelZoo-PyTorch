echo 'first we will call CycleGAN_AippTest.sh'
bash CycleGAN_AippTest.sh
echo 'Here we will compare the cosine similarity between the onnx format model and the om format model on the entire data set.'
python3 eval_acc.py --dataroot=./datasets/maps  --npu_bin_file=./result/dumpOutput_device0/
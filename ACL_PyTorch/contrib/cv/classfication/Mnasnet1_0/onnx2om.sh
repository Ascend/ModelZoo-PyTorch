for i in {1,4,8,16,32,64}
do
atc --framework=5 --model=mnasnet1.0.onnx --output=mnasnet1.0_bs"$i" --input_format=NCHW --input_shape="image:"$i",3,224,224" --log=debug --soc_version=Ascend710
done
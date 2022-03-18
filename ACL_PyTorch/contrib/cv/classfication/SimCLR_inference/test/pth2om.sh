rm -rf *.onnx
python3.7 Simclr_pth2onnx.py ../simclr.pth Simclr_model.onnx
rm -rf *.om
atc --framework=5 --model=Simclr_model.onnx --output=Simclr_model_bs1 --input_format=NCHW --input_shape="input:1,3,32,32" --log=info --soc_version=Ascend310 
if [ -f "Simclr_model_bs1.om" ]; then
    echo "Success changing pth to bs1_om."
else
    echo "Fail!"
fi
atc --framework=5 --model=Simclr_model.onnx --output=Simclr_model_bs16 --input_format=NCHW --input_shape="input:16,3,32,32" --log=info --soc_version=Ascend310
if [ -f "Simclr_model_bs16.om" ]; then
    echo "Success changing pth to bs16_om."
else
    echo "Fail!"
fi



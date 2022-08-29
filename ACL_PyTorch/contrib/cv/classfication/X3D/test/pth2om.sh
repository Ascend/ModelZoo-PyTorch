rm x3d_s.onnx
sed -i 's/ENABLE: True/ENABLE: False/g' SlowFast/configs/Kinetics/X3D_S.yaml
python3 X3d_pth2onnx.py --cfg SlowFast/configs/Kinetics/X3D_S.yaml X3D_PTH2ONNX.ENABLE True TEST.BATCH_SIZE 1 TEST.CHECKPOINT_FILE_PATH "x3d_s.pyth" X3D_PTH2ONNX.ONNX_OUTPUT_PATH "x3d_s.onnx"

rm x3d_s.om
source env.sh
atc --framework=5 --model=x3d_s.onnx --output=x3d_s1 --input_format=NCHW --input_shape="image:1,3,13,182,182" --log=error --soc_version=Ascend310 --precision_mode allow_mix_precision

atc --framework=5 --model=x3d_s.onnx --output=x3d_s16 --input_format=NCHW --input_shape="image:16,3,13,182,182" --log=error --soc_version=Ascend310 --precision_mode allow_mix_precision


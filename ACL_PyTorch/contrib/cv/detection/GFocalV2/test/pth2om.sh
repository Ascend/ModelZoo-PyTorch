rm -rf gfocal.onnx
python3.7 ./GFocalV2/tools/pytorch2onnx.py ./GFocalV2/configs/gfocal/gfocal_r50_fpn_1x.py ./gfocal_r50_fpn_1x.pth --output-file gfocal.onnx --input-img ./GFocalV2/demo/demo.jpg --shape 800 1216 --show
source env.sh
atc --framework=5 --model=./gfocal.onnx --output=gfocal_bs1 --input_format=NCHW --input_shape="input.1:1,3,800,1216" --log=debug --soc_version=Ascend310P3
if [ -f "gfocal_bs1.om" ]; then
    echo "success"
else
    echo "fail!"
fi

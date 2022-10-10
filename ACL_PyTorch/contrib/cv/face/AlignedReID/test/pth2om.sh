#参数1:pth文件路径 参数2:输出onnx文件路径 参数3:batchsize
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs32.onnx 32
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs16.onnx 16
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs8.onnx 8
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs4.onnx 4
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs1.onnx 1

#设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc --framework=5 --model=AlignedReID_bs1.onnx --output=AlignedReID_bs1 --input_format=NCHW --input_shape="image:1,3,256,128" --log=debug --soc_version=Ascend310 --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
atc --framework=5 --model=AlignedReID_bs4.onnx --output=AlignedReID_bs4 --input_format=NCHW --input_shape="image:4,3,256,128" --log=debug --soc_version=Ascend310 --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
atc --framework=5 --model=AlignedReID_bs8.onnx --output=AlignedReID_bs8 --input_format=NCHW --input_shape="image:8,3,256,128" --log=debug --soc_version=Ascend310 --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
atc --framework=5 --model=AlignedReID_bs16.onnx --output=AlignedReID_bs16 --input_format=NCHW --input_shape="image:16,3,256,128" --log=debug --soc_version=Ascend310 --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
atc --framework=5 --model=AlignedReID_bs32.onnx --output=AlignedReID_bs32 --input_format=NCHW --input_shape="image:32,3,256,128" --log=debug --soc_version=Ascend310 --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
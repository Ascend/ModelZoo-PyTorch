#!/bash
rm -rf *.onnx
python3.7.5 CSNLN_pth2onnx.py --n_feats 128 --pre_train model_x4.pt --save csnln_x4.onnx
python3.7.5 fix_onnx_prelu.py csnln_x4.onnx csnln_x4_fix.onnx
python3.7 -m onnxsim csnln_x4_fix.onnx csnln_x4_sim.onnx --input-shape=1,3,56,56
python3.7.5 perf_softmax_transpose.py csnln_x4_sim.onnx csnln_x4_perf.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf *.om
atc --framework=5 --model=csnln_x4_perf.onnx --output=csnln_x4_bs1 --input_format=NCHW --input_shape="input.1:1,3,56,56" --log=debug --soc_version=Ascend310
if [ -f "csnln_x4_bs1.om" ]; then
    echo "success"
else
    echo "fail"
fi

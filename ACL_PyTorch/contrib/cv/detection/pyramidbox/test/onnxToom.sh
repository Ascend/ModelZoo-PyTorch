 #!/bin/bash
cd ../
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=pyramidbox_1000.onnx --input_format=NCHW --input_shape="image:1,3,1000,1000" --output=pyramidbox_1000_bs1 --log=debug --soc_version=Ascend310 --precision_mode=force_fp32


source /usr/local/Ascend/ascend-toolkit/set_env.sh

export workdir=`pwd`
export modeldir=$workdir/SiamMask
export expdir=$modeldir/experiments/siammask_sharp

export PYTHONPATH=$PYTHONPATH:$modeldir:$expdir

cd $expdir
python3.7 $workdir/SiamMask_pth2onnx.py -type 0 --resume $workdir/SiamMask_VOT.pth --output_dir $workdir

cd $modeldir
patch -p1 < $workdir/SiamMask.patch

cd $expdir
python3.7 $workdir/SiamMask_pth2onnx.py  -type 1 --output_dir $workdir

cd $workdir
atc --framework=5 --model=mask.onnx --output=mask --input_format=NCHW --input_shape="search:1,3,255,255;template:1,3,127,127" --log=debug --soc_version=${chip_name} --out_nodes="Conv_310:0;Conv_327:0;Conv_344:0;Relu_154:0;Relu_187:0;Relu_229:0;Reshape_340:0"
atc --framework=5 --model=refine.onnx --output=refine --input_format=NCHW --input_shape="p3:1,256,1,1;p2:1,512,15,15;p1:1,256,31,31;p0:1,64,61,61" --log=debug --soc_version=${chip_name}
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export workdir=`pwd`
export modeldir=$workdir/SiamMask
export expdir=$modeldir/experiments/siammask_sharp

export PYTHONPATH=$PYTHONPATH:$modeldir:$expdir


cd $expdir
python3 $workdir/SiamMask_test.py --config config_vot.json --mask --refine --dataset VOT2016 --msame_path $workdir --save_path $workdir/om_io_files --om_path $workdir --device 0

echo "====accuracy data===="
cd $workdir
python3 SiamMask_eval.py --dataset VOT2016 --tracker_prefix C --result_dir $expdir/test/VOT2016

arch=`uname -m`
./benchmark.${arch} -round=10 -om_path=mask.om -device_id=0 -batch_size=1 > perf_mask.txt
./benchmark.${arch} -round=10 -om_path=refine.om -device_id=0 -batch_size=1 > perf_refine.txt

echo "====performance data===="
python3.7 test/parse.py perf_mask.txt perf_refine.txt

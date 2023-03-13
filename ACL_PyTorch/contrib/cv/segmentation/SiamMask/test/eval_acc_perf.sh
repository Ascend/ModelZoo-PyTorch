source /usr/local/Ascend/ascend-toolkit/set_env.sh

export workdir=`pwd`
export modeldir=$workdir/SiamMask
export expdir=$modeldir/experiments/siammask_sharp

export PYTHONPATH=$PYTHONPATH:$modeldir:$expdir


cd $expdir
python3 $workdir/SiamMask_test.py --config config_vot.json --mask --refine --dataset VOT2016 --om_path $workdir --device 0

echo "====accuracy data===="
cd $workdir
python3 SiamMask_eval.py --dataset VOT2016 --tracker_prefix C --result_dir $expdir/test/VOT2016

python3 -m ais_bench --loop=1000 --model=mask.om --device=0 --batchsize=1 > perf_mask.txt
python3 -m ais_bench --loop=1000 --model=refine.om --device=0 --batchsize=1 > perf_refine.txt

echo "====performance data===="
python3.7 test/parse.py perf_mask.txt perf_refine.txt

source env_npu.sh
DATE_TIME=`date +'%Y_%m_%d_%H_%M_%S'`
mkdir ./../logs
export LD_PRELOAD={sklearn-path}/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
echo ./../logs/run_16p_$DATE_TIME.log
python ./deepy.py train.py -d configs 20B_16p.yml >  ./../logs/run_16p_$DATE_TIME.log  2>&1 &

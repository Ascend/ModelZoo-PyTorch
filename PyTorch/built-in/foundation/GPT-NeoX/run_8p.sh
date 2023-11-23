source env_npu.sh
DATE_TIME=`date +'%Y_%m_%d_%H_%M_%S'`
export LD_PRELOAD={sklearn-path}/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
mkdir ./../logs
echo ./../logs/run_8p_$DATE_TIME.log
python3 ./deepy.py train.py -d configs 20B_8p.yml >  ./../logs/run_8p_$DATE_TIME.log  2>&1 &


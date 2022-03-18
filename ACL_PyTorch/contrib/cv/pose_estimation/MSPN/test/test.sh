export MSPN_HOME=$(pwd)
export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
python $MSPN_HOME/exps/mspn.2xstg.coco/config.py -log

datasets_path="$MSPN_HOME/dataset/COCO"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# arch=`uname -m`

echo "Preprocessing data ..."

python3.7 MSPN_preprocess.py -d ${datasets_path}

python gen_dataset_info.py  ./pre_dataset ./image.info 256 192

source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=MSPN_bs1.om -input_text_path=./image.info -input_width=192 -input_height=256 -output_binary=True -useDvpp=False

#./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=MSPN_bs16.om -input_text_path=./image.info -input_width=192 -input_height=256 -output_binary=True -useDvpp=False

python MSPN_postprocess.py

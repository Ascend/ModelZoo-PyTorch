echo "=======get dataset======="
rm -rf dataset
python3.7 c51_get_dataset.py c51.model c51.stats  dataset/states dataset/actions 1000
python3.7 get_dataset_bin.py dataset/states dataset/bin dataset/out
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

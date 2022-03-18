
echo "=======get dataset======="
rm -rf dataset
python3.7 get_dataset.py  --pth-path='weight_pth' --state-path='state_path' 1000
python3.7 get_dataset_bin.py --state-path='state_path' --bin-path='bin_path'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


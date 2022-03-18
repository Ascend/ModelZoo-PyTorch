source env.sh
export PYTHONPATH=/usr/lib:$PYTHONPATH
rm -rf ./runs

python3.7 -u train_brats2018_new.py --amp --world_size 1 --rank 0 --prof
data_path="/home/CSNLN"
for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done
nohup python3 main.py \
	--epochs 2 \
	--model CSNLN \
	--data_test Set5 \
	--dir_data ${data_path} \
	--scale 2 \
	--n_feats 128 \
	--depth 12 \
	--rank_id 2 \
	--chop \
	--batch_size 16 \
	--patch_size 96 \
	--save CSNLN_x2 \
	--data_train DIV2K \
	--save_models \
	--n_threads 18 \
	--distributed 0 \
	--n_GPUs 1 \
	--print_every 1 \
	--seed 12 \
	--amp \
	--lr 0.0001 \
	--performance &

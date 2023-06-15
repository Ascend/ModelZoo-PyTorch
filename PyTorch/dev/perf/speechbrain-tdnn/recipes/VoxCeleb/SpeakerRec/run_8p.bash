export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12581"

RANK_ID_START=0

for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE+RANK_ID_START));RANK_ID++))
do
    echo ${RANK_ID}
    export RANK=${RANK_ID}
    KERNEL_NUM=$(($(nproc)/8))
    PID_START=$((KERNEL_NUM))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    nohup taskset -c $PID_START-$PID_END python3 train_speaker_embeddings.py --distributed_launch --distributed_backend='nccl' --local_rank=$RANK_ID hparams/train_ecapa_tdnn.yaml --data_folder=/data/voxceleb/ > train_$RANK_ID.log 2>&1 &
done

wait 

cur_path=`pwd`
file=`find $cur_path -name CKPT+*`
conf_file=$cur_path/hparams/verification_ecapa.yaml
pretrain_path=`cat $conf_file | grep pretrain_path:`
sed -i "s|$pretrain_path|pretrain_path: $file|g" $conf_file

cd $cur_path/results/voxceleb1_2/speaker_verification_ecapa_big_vox2only/save/

ln -s -f $file/embedding_model.ckpt embedding_model.ckpt

cd ../../../../

nohup python3 speaker_verification_cosine.py hparams/verification_ecapa.yaml --device npu --data_folder=/data/voxceleb/ &
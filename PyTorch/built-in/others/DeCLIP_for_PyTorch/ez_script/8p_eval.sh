source test/env_npu.sh

export PYTHONPATH=./DeCLIP:$PYTHONPATH

export SLURM_NTASKS=8
export SLURM_NODELIST="        127.0.0.1"
export MASTER_PORT="23333"

RANK_ID_START=0
RANK_SIZE=$SLURM_NTASKS

KERNEL_NUM=$(($(nproc) / 8))
for ((RANK_ID = $RANK_ID_START; RANK_ID < $((RANK_SIZE + RANK_ID_START)); RANK_ID++)); do
	PID_START=$((KERNEL_NUM * $RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))

	export SLURM_PROCID=$RANK_ID
	nohup taskset -c $PID_START-$PID_END python -u runner.py --config config.yaml --evaluate &

done

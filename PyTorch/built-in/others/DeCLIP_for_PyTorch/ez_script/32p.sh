source test/env_npu.sh
export PYTHONPATH=./DeCLIP:$PYTHONPATH

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600

# 运行前需要配置
export HCCL_IF_IP=xxxx # 当前节点IP，可以通过ifconfig或$(hostname -I)获取。由于网卡配置的多样性，此处建议手动设置
MASTER_IP=xxxx # 设置主节点IP，每个机器共用一个主节点
export NNODE=0 # 主节点设置0，其他节点依次设置，如1,2,3

export SLURM_NTASKS=32
export SLURM_NODELIST="        $MASTER_IP"
export MASTER_PORT="23333"

KERNEL_NUM=$(($(nproc) / 8))
for ((RANK_ID = 0; RANK_ID < 8; RANK_ID++)); do
	PID_START=$((KERNEL_NUM * $RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))

	export SLURM_PROCID=$(($NNODE * 8 + RANK_ID))
	nohup taskset -c $PID_START-$PID_END python -u runner.py --config config.yaml &

done

wait

if [ $NNODE == 0 ];then
	bash ez_script/8p_eval.sh
fi

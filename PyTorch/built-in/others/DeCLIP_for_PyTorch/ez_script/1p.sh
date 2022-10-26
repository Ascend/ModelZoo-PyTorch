source test/env_npu.sh

export PYTHONPATH=./DeCLIP:$PYTHONPATH

export SLURM_NTASKS=1
export SLURM_NODELIST="        127.0.0.1"
export MASTER_PORT="23333"
export SLURM_PROCID=0

python -u runner.py --config config.yaml


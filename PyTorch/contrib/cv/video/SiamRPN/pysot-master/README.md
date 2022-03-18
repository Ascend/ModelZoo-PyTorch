# SiamRPN

## Requirements

- `pip install -r requirements.txt`


## Training
The address of the dataset required for training can be configured in ./experiments/siamrpn_r50_l234_dwxcorr_8gpu

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh

# training 1p performance
bash ./test/train_performance_1p.sh

# training 8p accuracy
bash ./test/train_full_8p.sh

# training 8p performance
bash ./test/train_performance_8p.sh


Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/train_per_${device_id}.log   # training performance result log
    test/output/devie_id/SiamRPN_${RANK_SIZE}p_acc.log   # training accuracy result log

NPU8P:
e19 A:0.636 R:0.228 EAO:0.429
e20 A:0.638 R:0.247 EAO:0.405

NPU1P:
e19 A:0.642 R:0.261 EAO:0.388
e20 A:0.642 R:0.280 EAO:0.365

# Experiment all tricks without center loss with re-ranking : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# with re-ranking
data_path=""
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
python3 tools/finetune.py \
	--config_file='configs/softmax_triplet_with_center.yml' \
	--npus=8 \
	--loss_scale="64.0" \
	DATASETS.NAMES "('market1501')" \
	DATASETS.ROOT_DIR "('${data_path}')" \
	TEST.RE_RANKING "('yes')" \
	MODEL.PRETRAIN_CHOICE "('self')" \
	TEST.WEIGHT "('./logs/E-a-t-m-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on/8p/resnet50_model_120.pth')"

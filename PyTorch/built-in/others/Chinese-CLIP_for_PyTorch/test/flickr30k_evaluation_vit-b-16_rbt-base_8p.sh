beginTime=`date "+%Y-%m-%d %H:%M:%S"`

export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

DATAPATH=${1}

split=test # 指定计算valid或test集特征
resume=${DATAPATH}/experiments/flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu/checkpoints/epoch_latest.pt
dataset_name=Flickr30k-CN

# 图文特征提取
# 产出图文特征默认保存于${DATAPATH}/datasets/${dataset_name}目录下
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese

# KNN检索
# 文到图检索
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"

# 图到文检索
python -u cn_clip/eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"

# Recall计算
# 文到图检索
python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
echo "文到图检索"
cat output.json

# 图到文检索
python cn_clip/eval/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

python cn_clip/eval/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    output_tr.json
echo "图到文检索"
cat output_tr.json

endTime=`date "+%Y-%m-%d %H:%M:%S"`
duration=$(($(date +%s -d "${endTime}")-$(date +%s -d "${beginTime}")))
hour=$(( $duration/3600 ))
min=$(( ($duration-${hour}*3600)/60 ))
sec=$(( $duration-${hour}*3600-${min}*60 ))
HMS=`echo ${hour}:${min}:${sec}`
echo "开始：$beginTime"
echo "结束：$endTime"
echo "耗时：$HMS"
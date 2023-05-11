# 数据集路径,保持为空,不需要修改
data_path=""
output_path="./output"
pth_path="./R-101.pkl"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    elif [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

export DETECTRON2_DATASETS=${data_path}
python3 -u projects/PointRend/train_net.py \
         --config-file projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml \
         --device-ids 0 1 2 3 4 5 6 7 \
         --num-gpus 8 \
         --eval-only \
         MODEL.WEIGHTS ${pth_path} \
         OUTPUT_DIR ${output_path}
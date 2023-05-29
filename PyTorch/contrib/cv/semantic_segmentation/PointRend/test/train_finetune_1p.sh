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
         --solver-steps 40000 55000 \
         --num-gpus 1 \
         MODEL.DEVICE npu:0 \
         MODEL.WEIGHTS ${pth_path} \
         MODEL.POINT_HEAD.NUM_CLASSES 19 \
         MODEL.SEM_SEG_HEAD.NUM_CLASSES 19 \
         SEED 1234 \
         SOLVER.BASE_LR 0.005 \
         SOLVER.IMS_PER_BATCH 16 \
         SOLVER.WARMUP_ITERS 1000 \
         SOLVER.MAX_ITER 65000 \
         AMP.ENABLED 1 \
         AMP.LOSS_SCALE 512 \
         TEST.ENABLED 0 \
         DATALOADER.NUM_WORKERS 24 \
         OUTPUT_DIR ${output_path} 
bash test/train_eval_8p.sh --data_path=${data_path} --output_path=${output_path} --pth_path=${output_path}"/model_final.pth"
##### 参数设置
version=${1:-"5.0"}
model=${2:-"yolov5s"}
bs=${3:-"4"}
type=${4:-"fp16"}
mode=${5:-"infer"}
output_dir=${6:-"output"}
soc=${7:-"Ascend710"}
install_path=${8:-"/usr/local/Ascend/ascend-toolkit"}
arch=${7:-"x86_64"}

## 设置环境变量
source ${install_path}/set_env.sh

## pt导出om模型
bash common/pth2om.sh --version $version \
                      --model $model \
                      --bs $bs \
                      --type $type \
                      --mode $mode \
                      --output_dir $output_dir \
                      --soc $soc

if [ $? -ne 0 ]; then
    echo -e "pth导出om模型 Failed \n"
    exit 1
fi

## 推理om模型
bash common/eval.sh --version $version \
                    --model $model \
                    --bs $bs \
                    --type $type \
                    --mode $mode \
                    --output_dir $output_dir \
                    --install_path $install_path \
                    --arch $arch


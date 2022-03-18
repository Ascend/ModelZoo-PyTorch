# C3D模型 说明文档

note
- please prepare dataset follow https://github.com/open-mmlab/mmaction2/blob/master/docs/data_preparation.md

## Requirements
安装NPU配套的run包、apex（version：ascend）、torch（version:ascend）外，另需执行以下shell
```shell
pip3 install torchvision==0.2.2.post3
cd $mmaction2
pip3 install -r requirements.txt
```
安装 mmcv
```
export GIT_SSL_NO_VERIFY=1
git config --global http.sslVerify false
git clone -b v1.3.9 --depth=1 https://github.com/open-mmlab/mmcv.git
source ./env_npu.sh; cd ${curPath}/mmcv; python3.7 setup.py build_ext; python3.7 setup.py develop
```
## 修改mmcv及apex底层依赖环境
apex修改地址 `${package_dir}=../lib/python3.7/site-packages/apex/amp/` \
`${curPath}` 指 当前工程mmaction2目录
```shell
# mmcv_need
cp ${curPath}/additional_need/mmcv/distributed.py   ${curPath}/mmcv/mmcv/parallel/
cp ${curPath}/additional_need/mmcv/test.py   ${curPath}/mmcv/mmcv/engine/
cp ${curPath}/additional_need/mmcv/dist_utils.py   ${curPath}/mmcv/mmcv/runner/
cp ${curPath}/additional_need/mmcv/optimizer.py  ${curPath}/mmcv/mmcv/runner/hooks/
cp ${curPath}/additional_need/mmcv/epoch_based_runner.py  ${curPath}/mmcv/mmcv/runner/

# apex_need
cp ${curPath}/additional_need/amp/scaler.py ${package_dir}/apex/amp/
```


## Dataset Prepare

### Step 1. Prepare Videos

将处理好的数据集放到数据目录`data_dir = $MMACTION2/data`文件夹下解压

### Step 2. Check Directory Structure

确认最终的目录结构是否是如下格式。

```
mmaction2
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05
```



## 模型训练
以下脚本须在工程根目录`$MMACTION2`下执行：
### 单卡训练

```shell
sh ./test/run_1p.sh
```
### 单卡性能测试

```shell
sh ./test/run_1p_perf.sh
```
### 单卡精度测试
首先查看训练输出的best_top1_acc_epoch_*.pth
```shell
cd ./work_dirs/npu_1p
ls -all
```
修改脚本参数.
```shell
vim ./test/test_1p.sh
```
将latest.pth 替换为work_dir目录下的best_top1_acc_epoch_*.pth，修改完毕后，保存退出，执行对应测试脚本
```shell
sh ./test/test_1p.sh
```
### 8卡训练

```shell
chmod a+x ./tools/dist_train.sh
sh ./test/run_8p.sh
```
### 8卡性能测试

```shell
chmod a+x ./tools/dist_train.sh
sh ./test/run_8p_perf.sh
```
### 8卡精度测试
与单p的精度测试相似，首先查看训练输出的best_top1_acc_epoch_*.pth
```shell
cd ./work_dirs/npu_8p
ls -all
```
修改脚本参数
```shell
vim $mmaction2/test/test_8p.sh
```
将latest.pth 替换为work_dir目录下的best_top1_acc_epoch_*.pth，修改完毕后，保存退出，执行对应测试脚本
```shell
sh ./test/test_8p.sh
```
## log文件地址
运行产生的log文件在 `$MMACTION2/work_dirs` 文件夹下


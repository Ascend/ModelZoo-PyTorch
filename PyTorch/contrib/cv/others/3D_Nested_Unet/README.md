# 3D_Nested_Unet

This implements training of 3D_Nested_Unet on the MSD dataset, mainly modified from [pytorch/examples](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch).

## 3D_Nested_Unet Detail

UNet++ differs from the original U-Net in three ways: 1) having convolution layers on skip pathways, which bridges the semantic gap between encoder and decoder feature maps; 2) having dense skip connections on skip pathways, which improves gradient flow; and 3) having deep supervision, which enables model pruning and improves or in the worst case achieves comparable performance to using only one loss layer.

**本教程的文件及其说明**
```
程序及脚本文件
├── change_bs.py               //修改batchsize的程序
├── get_dice_result.py         //读取dice结果的程序
test文件夹
├── env_npu.sh                 //NPU环境变量
├── train_full_1p.sh           //NPU 1P训练脚本
├── train_full_8p.sh           //NPU 8P训练脚本
├── train_performance_1p.sh    //NPU 1P性能测试脚本
├── train_performance_8p.sh    //NPU 8P性能测试脚本
其他文件
├── README.md                  //本文的上手指导
├── new_npu.patch              //修改源代码的补丁（修改为NPU版本）
├── new_gpu.patch              //修改源代码的补丁（修改为GPU版本，与本文教程无关）
├── requirements.txt           //NPU版本的环境依赖，由pip freeze > requirements.txt生成
├── requirements_gpu.txt       //GPU版本的环境依赖
其他附件（不在本代码仓中获得）
├── v100_1p.log                //GPU 1P训练日志
├── v100_8p.log                //GPU 8P训练日志
├── 910A_1p.log                //NPU 1P训练日志
├── 910A_8p.log                //NPU 8P训练日志
├── v100_1p.prof               //GPU 1P prof文件
├── 910A_1p.prof               //NPU 1P prof文件
└── gpu_code.tar               //GPU 1P及GPU 8P训练代码
```
**关键环境：**
| 依赖名 | 版本号 |
| :------: | :------: |
| CANN | 5.1.RC1 |
| python | 3.7.5 |
| apex | ==0.1+ascend.20220315 |
| torch | ==1.5.0+ascend.post5.20220315 |
| batchgenerators | ==0.21 |
| 其他依赖 | 可在后文实验步骤中查阅 |

**相关链接：**
| 名称和地址 | 说明 |
| :------: | :------: |
| [UNET官方代码仓](https://github.com/MIC-DKFZ/nnUNet)  | UNET官方框架。 |
| [UNET++官方代码仓](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch)  | 依据UNET官方框架进行开发的UNET++官方代码。 |
| [MSD数据集（Medical Segmentation Decathlon）](http://medicaldecathlon.com/)  | 医学十项全能数据集，内含10个子任务，本文只对任务3肝脏任务进行验证。数据图像均为三维灰度图像，您可以下载使用ITK-SNAP工具来可视化图像。 |
| [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)  | 三维图像可视化工具。 |

## 1 环境准备

### 1.1 获取源代码
下载官方代码仓，并退回至指定版本，以保证代码稳定不变。
```
cd /home/hyp/  # 以/home/hyp/为工程目录为例
git clone https://github.com/MrGiovanni/UNetPlusPlus.git
cd UNetPlusPlus
git reset e145ba63862982bf1099cf2ec11d5466b434ae0b --hard
```

### 1.2 安装依赖，修改模型代码
在执行下述命令前，请确保已经手动安装了NPU版本的torch和apex，否则将会安装CPU版本的torch。
```
cd /home/hyp/UNetPlusPlus/
patch -p1 < ../new_npu.patch  # 载入代码修改补丁
cd pytorch
pip install -e .
pip install batchgenerators==0.21  # 该依赖十分关键，指定其版本再手动安装一次

# 您也可以通过requirements来安装依赖包，但我们不推荐该方法
pip install -r requirements.txt
```
patch命令的最后一个参数需要指定本仓中的new_npu.patch文件的路径。由于该模型需要将命令注册到环境中才能找到正确的函数入口，所以我们仍然需要一步pip来将代码注册到环境中。除此之外，每次将代码文件进行大幅度地增减时，“pip install -e .”都是必须的，否则很可能出现“import nnunet”错误。

我们不推荐您使用requirements.txt的方式来安装环境，因为这很可能遗漏nnunet的注册步骤，使得后续实验无法进行。

注：如果在执行“pip install -e .”或在后面的实验过程中，仍然出现了环境包或模块的安装或导入错误，则很可能需要重新手动安装部分包。我们认为，原作者没有完全指明必要的依赖，而那些隐藏的依赖目前已经升级了多个版本，导致各个依赖间的关系出现变化，进而使得如今完全按作者的描述安装依赖是不可行的。我们在多个服务器上，已观测到仍然可能出现异常的包有但不仅限于：
 - apex (NPU版本)
 - torch (NPU版本)
 - decorator
 - sympy
 - SimpleITK
 - matplotlib
 - batchgenerators==0.21
 - pandas
 - scikit-image
 - sklearn
 - nibabel
 
在多个不同的服务器环境上，上述的包都有过至少两次安装失败的经历。通常手动安装上述的包就可以解决问题，使用诸如“pip install batchgenerators==0.21”的方式来重新安装界面报错提示中指定的那些包，或更换镜像源。第二种方法是使用离线whl包进行安装。若仍无法解决，则很可能是系统底层版本过低，例如GLIBC。

 您可以使用如下命令来确认自己的系统是x86架构还是aarch架构。
```
uname -a
```

### 1.3 设置nnunet环境变量
参照UNET的描述，在硬盘空间充足的路径下，我们以/home/hyp/为例，在该路径下创建一个新的文件夹environment，用于存放相关实验数据，该路径不强求和项目所在路径相同。在environment中再创建三个子文件夹：nnUNet_raw_data_base、nnUNet_preprocessed和RESULTS_FOLDER。这三个文件夹不强求位于同一目录下，甚至可以位于多块硬盘下，出于检索方便的考虑，我们推荐将其位于同一目录下，例如environment。我们推荐您至少确保该路径下（指代environment）有400GB的存储空间。
```
cd environment
mkdir nnUNet_raw_data_base
mkdir nnUNet_preprocessed
mkdir RESULTS_FOLDER
```
最后修改/root/.bashrc文件，在文件尾部添加如下环境变量。这样以后每次开启新会话时，位于.bashrc中的环境变量都会自动导入，无需用户再手动export一次。
```
export nnUNet_raw_data_base="/home/hyp/environment/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/hyp/environment/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/hyp/environment/RESULTS_FOLDER"
```
使用source命令来刷新环境变量。如果您实在不想修改.bashrc文件，您也可以在您当前会话中直接输入上面的三条export语句来设置环境变量，但是这些变量只在当前会话内有效。
```
source ~/.bashrc
```
注：我们十分推荐将以上文件夹置于SSD上。如果使用的是机械硬盘，我们观察到该模型会占据大量的IO资源，导致系统卡顿。如果您还希望使用您设备上可用的GPU，则还需要额外添加以下环境变量。
```
# 配置GPU编号为0至3的四卡环境
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
最后别忘了，在NPU上进行的任何实验，都需要载入NPU环境变量。命令大概如下，请用户自行解决NPU环境变量的载入。如果出现了HCCL错误，则很可能是没有载入正确的NPU环境变量。
```
例1：source env_npu.sh
例2：source set_env.sh
```

### 1.4 获取数据集
获取[Medical Segmentation Decathlon](http://medicaldecathlon.com/)，下载其中的第三个子任务集Task03_Liver.tar，放到environment目录下（后文中environment均指代/home/hyp/environment/），并解压。该数据集中的Task03_Liver在后续实验过程中会被裁剪展开，该数据集在使用时将占用约260GB的存储空间。
```
# 确认系统剩余存储空间
df -h
# 转移并解压数据集
mv ./Task03_Liver.tar /home/hyp/environment/
cd /home/hyp/environment/
tar xvf Task03_Liver.tar
```
至此，在environment文件夹内，文件结构应像如下所示，它们与上一节在.bashrc中设置的环境变量路径要保持相同：
```
environment/
├── nnUNet_preprocessed/
├── nnUNet_raw_data_base/
├── RESULTS_FOLDER/
├── Task03_Liver/
└── Task03_Liver.tar
```

### 1.5 数据格式转换
在environment文件夹内，使用nnunet的脚本命令，对解压出的Task03_Liver文件夹中的数据进行数据格式转换。该脚本将运行约5分钟，转换结果将出现在nnUNet_raw_data_base子文件夹中。
```
nnUNet_convert_decathlon_task -i Task03_Liver -p 8
```
如果您的设备性能较差或者该命令在较长时间后都未结束，您可以将参数-p的数值调小，这将消耗更多的时间。
```
nnUNet_convert_decathlon_task -i Task03_Liver -p 1
```
注：若您在之后的实验过程中想要重置实验或者数据集发生严重问题（例如读取数据时遇到了EOF等读写错误），您可以将nnUNet_preprocessed、nnUNet_raw_data_base和RESULTS_FOLDER下的文件全部删除，并重新完成本节及后续过程。

### 1.6 实验计划与预处理
nnunet十分依赖数据集，这一步需要提取数据集的属性，例如图像大小、体素间距等，并生成后续实验的配置文件。若删减数据集图像，都将使后续实验配置发生变化。使用nnunet的脚本命令，对nnUNet_raw_data_base中的003任务采集信息。这个过程将持续半小时至六小时不等，具体时间依赖于设备性能，转换结果将出现在nnUNet_preprocessed子文件夹中。
```
nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity
```
我们观察到，该过程很可能意外中断却不给予用户提示信息，这在系统内存较小时会随机发生，请您确保该实验过程可以正常结束。如果您的设备性能较差或者较长时间后都未能正常结束，您可以改用下面的命令来降低系统占用，而这将显著提升该步骤的运行时间。实践来看，通过输入free -m命令，如果系统显示的available Mem低于30000或在30000左右，则我们推荐您改用下面的命令。
```
nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity -tl 1 -tf 1
```
注：若在后续的实验步骤中出现形如“RuntimeError: Expected index [2, 1, 128, 128, 128] to be smaller than self [2, 3, 8, 8, 8] apart from dimension 1”的错误，请删除environment/nnUNet_preprocessed/Task003_Liver/以及environment/nnUNet_raw_data_base/nnUNet_cropped_data/下的所有文件，然后重新完成本节内容。

同时，本节内容也划分出了训练集、验证集和测试集。在environment/nnUNet_preprocessed/Task003_Liver/dataset.json中记录了训练集和测试集的划分，而在相关.pkl中存储了训练集和验证集的划分。本模型采取了五折交叉验证，受限于部分因素，我们暂且只对第1折实验来讲解。对于第1折实验，其验证集的划分如下：
```
第1折（fold 0）全部验证集图像的编号（共27张，从小到大排序，实际程序读取顺序不明确）：
3, 5, 11, 12, 17, 19, 24, 25, 27, 38, 40, 41, 42, 44, 51, 52, 58, 64, 70, 75, 77, 82, 101, 112, 115, 120, 128
```
验证集图像自然不参与训练，但是是从训练集中划分出来的，上述编号的验证集图像的原始图像均来自训练集图像所在目录：environment/nnUNet_raw_data_base/nnUNet_raw_data/Task003_Liver/imagesTr/liver_XXX_0000.nii.gz，其中XXX即为图像编号。

## 2 模型训练

### 2.1 关于修改batchsize的方法
在完成1.6节时，UNET++模型中关于batchsize的设定便存储在了environment/nnUNet_preprocessed/Task003_Liver/nnUNetPlansv2.1_plans_3D.pkl中，默认值为2。使用本文提供的change_bs.py程序可以将其中的batchsize修改为您想要的数值，第一个参数-path接受文件夹nnUNet_preprocessed的所在路径，第二个参数-size接受修改后的batchsize值，将batchsize修改为8的示例如下：
```
python change_bs.py -path /home/heyupeng/environment/ -size 8
# nnUNet_preprocessed文件夹位于/home/heyupeng/environment/nnUNet_preprocessed/
```
注：本实验的数据量很大，1张图像将使用约8-10GB的显存空间。因此通常只能将batchsize设置为显卡数量的1或2倍值，而显卡数量取决于将要使用的是单卡训练还是多卡训练。

### 2.2 NPU 1P训练
使用2.1节的内容，**将batchsize修改为1或2**（对应于拥有NPU 10GB显存空间或NPU 20GB显存空间）。

UNET++模型的1P训练启动命令为：
```
# 方法一：使用nnUNet_train命令启动单卡训练
nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 Task003_Liver 0
```
其中命令nnUNet_train代表使用nnUNet框架进行训练，这意味着你可以在任意目录下来随时启动模型训练。本质上，该命令会执行UNetPlusPlus/pytorch/nnunet/run/run_training.py，并传递参数。参数3d_fullres指明使用3D全分辨率，参数nnUNetPlusPlusTrainerV2指明了使用的模型训练器，参数Task003_Liver指明了使用数据集003，最后的参数0代表本次为第0折验证（总共有5个交叉验证实验），这些设置全部遵循了UNET++的官方设置。具体的参数说明，可以在UNET官方代码仓中找到说明。如果想完全复现UNET++全部实验，你还需要将最后的参数设置分别设置为0、1、2、3和4，进行总计五次独立的交叉实验。

如果想查看最后的训练结果和精度，请查阅2.4节。您也可以直接使用下面的脚本来一次性完成两个操作，即启动NPU 1P训练和查看精度。
```
# 方法二：脚本启动训练，训练结束后会输出精度
bash test/train_full_1p.sh -path /home/heyupeng/environment/
```

### 2.3 NPU 8P训练
使用2.1节的内容，**将batchsize修改为8或16**（对应于拥有8块NPU 10GB显存空间或8块NPU 20GB显存空间）。

UNET++模型的8P训练启动命令为：
```
# 方法一：使用pytorch的DDP方式启动多卡训练
cd UNetPlusPlus/pytorch/nnunet/run
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 0 --dbs
# 其中的run_training_DDP.py位于项目目录UNetPlusPlus/pytorch/nnunet/run/下。
```
多卡训练不再以nnUNet_train命令作为启动方式，而是直接调用多卡运行程序run_training_DDP.py来启动，所以您需要事先找到项目代码UNetPlusPlus/pytorch/nnunet/run/run_training_DDP.py的路径。参数master_port指明了通信端口号，参数nproc_per_node指明了使用的多卡的卡数量，参数003指明同样使用数据集003，参数0代表本次为第0折验证（总共有5个交叉验证实验），最后的参数--dbs则指明我们的batchsize需要平分到所有的设备上。以batchsize=16 nproc_per_node=8为例，则说明每个设备上的batchsize为16/8=2。关于多卡训练的更多解释，请查阅2.7节内容。

如果想查看最后的训练结果和精度，请查阅2.4节。您也可以直接使用下面的脚本来一次性完成两个操作，即启动NPU 8P训练和查看精度。
```
# 方法二：脚本启动训练，训练结束后会输出精度
# 先切换至项目目录UNetPlusPlus的所在路径下
cd /home/heyupeng/environment/
bash test/train_full_8p.sh -path /home/heyupeng/environment/
```
注：多卡代码同样是试验性的，在训练过程中，用户可以自行提前终止训练。但是如果运行过程中出现了异常报错，这将会有很大可能会出现僵尸进程，占据着NPU的显存空间和master_port=1234，使得用户受限于显存空间不足或端口号被占用而无法开启新的多卡实验。对于第一种情况，通过使用"npu-smi info"命令来观察最后一列的数值，通常表现为"20000 / 32768"。对于第二种情况，使用"ps -aux | grep python"或"ps -aux | grep nnUNet"，观察是否存在大量的进程残留。遗憾的是，有时我们无法通过"kill"命令来杀死这些进程或其父进程，只能采用"reboot"命令来重启设备。

### 2.4 查看精度结果与模型权重
模型训练结束后，会立即开始对验证集的推理，并对全部结果做精度验证。最终实验的精度将记录在environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/XXX/fold_0/validation_raw/summary.json中，其中的XXX可能为nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1或nnUNetPlusPlusTrainerV2_hypDDP__nnUNetPlansv2.1，分别对应于NPU 1P和NPU 8P。结果存在浮动是正常现象，您可以参照如下的结构树来找到其中的Dice指标：
```
summary.json
├── "author"
├── "description"
├── "id"
├── "name"
├── "results"
│   ├── "all"
│   └── "mean"
│       ├── "0"
│       ├── "1"
│       │   ├── ...
│       │   ├── "Dice": 0.9667251131239053  # 即为Liver 1_Dice
│       │   └── ...
│       └── "2"
│           ├── ...
│           ├── "Dice": 0.7141959958916767  # 即为Liver 2_Dice
│           └── ...
├── "task": "Task003_Liver"
└── "timestamp"
```
如果您在训练时，采用了

这是在第0折交叉验证下的结果，验证集图像只有27张，本文的肝脏数据是在不同的实验仪器下采集的，图像尺寸与图像质量均存在较大差异。选用不同的交叉必然会导致不同的实验结果，但对精度达标的目标来说影响不大。由于单次训练预计耗时3天或更久，完成5次交叉的实验周期过长，所以我们全程只使用了fold 0，即第一折交叉验证。当然，直接使用作者提供的其他折实验模型权重并作验证集测试，也能达到较好的精度。

您也可以通过使用本文提供的get_dice_result.py程序来快速获取summary.json中的dice结果，第一个参数-path接受文件夹RESULTS_FOLDER的所在路径，第二个参数-mode接受想要查看哪个实验的结果，将batchsize修改为8的示例如下：
```
python get_dice_result.py -path /home/heyupeng/environment/ -mode 1p
# RESULTS_FOLDER文件夹位于/home/heyupeng/environment/RESULTS_FOLDER/
# 参数--mode只接受两种取值，分别是1p和8p，对应于前文提到的NPU 1P和NPU 8P。
```

而权重文件则位于environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/XXX/fold_0/下，文件后缀名为(.modl，.model.pkl)的一对文件，前者为模型权重，后者为模型的其他参数记录。如果您想载入新的权重文件，建议您将这一对文件一并进行拷贝。

### 2.5 NPU 1P性能
使用前面提到的nnUNet_train命令也可以开启性能测试，但是这次训练不再是完整的训练过程，而是只运行几个step的短过程。添加额外的参数--other_use，可以设置为：
 - fps（常用）：获取性能（FPS）结果的运行方式。程序将先运行10个step，之后再运行100个step，并测100个step的平均运行时间。
 - prof（可选）：获取单P prof文件的运行方式。程序将先运行5个step，之后以torch.autograd.profiler.profile()的方式来收集Pytorch api耗时情况。运行结束后，会在当前路径下生成output.prof文件。获取NPU prof时，需要添加环境变量设置export ASCEND_GLOBAL_LOG_LEVEL=3和export TASK_QUEUE_ENABLE=1。
 
 我们推荐您将-fold参数修改为非0值，这样性能测试的结果不会覆盖您原有的训练结果上，运行命令示例如下：
```
# 注意，我们将倒数第二个参数的值从0修改为了1，这样便不会覆盖RESULT_FOLDER中fold 0的结果文件
nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 Task003_Liver 1 --other_use fps
```
您同样可以直接使用下面的脚本来获取性能结果。
```
bash test/train_performance_1p.sh
```

### 2.6 NPU 8P性能
和上一节一样，使用下面的命令开启多卡性能测试：
```
# 注意，我们将倒数第二个参数的值从0修改为了1，这样便不会覆盖RESULT_FOLDER中fold 0的结果文件
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 1 --dbs --other_use fps
# 其中的run_training_DDP.py位于项目目录UNetPlusPlus/pytorch/nnunet/run/下。
```
您同样可以直接使用下面的脚本来获取性能结果。
```
# 先切换至项目目录UNetPlusPlus的所在路径下
cd /home/heyupeng/environment/
bash test/train_performance_8p.sh
```
注：测试8P性能时，程序只会输出每张卡上的结果。因此得到的也是8P情况下单卡的性能，并且结果会输出8次。您需要将结果乘以8，以得到真实的8P性能。8P请勿将--other_use设置为prof。后续我们可能会更新这部分代码，使得8p仅输出一次结果，不需要用户手动乘以8。

### 2.7 GPU 8P训练（可选）
UNET++官方只提供了GPU单卡的训练代码，未提供GPU多卡的代码。而UNET官方则是提供了试验阶段的GPU多卡代码。因此本文以UNET为基准，参考了UNET的1P->8p的修改记录和训练流程，进而补写了UNET++的8P代码。

但是本教程只负责NPU的训练内容，如果读者想要尝试进行GPU 8P的训练，请使用本教程中提供的另外一份补丁包new_gpu.patch，重复上述所有实验步骤。其中环境依赖中的torch>=1.6.0，并且不需要额外安装apex。同时，对于1.3节的内容，需要额外补充一个新的环境变量：
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 指明了可用的GPU设备的编号为0至7
```
最后使用如下命令，便可以开启GPU 8P的训练：
```
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=8 run/run_training_DDP.py 3d_fullres nnUNetPlusPlusTrainerV2_hypDDP 003 0 --dbs
# 其中的run_training_DDP.py位于项目目录UNetPlusPlus/pytorch/nnunet/run/下。
```
参数nproc_per_node指明了使用的GPU卡数，通常设置为2/4/8。最终的GPU 8P的训练结果与作者提供的曲线图几乎吻合，而倘若只使用作者提供的GPU 1P代码，会观察到模型无法收敛。你可以在environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/XXX/fold_0/下找到progress.png图像，即为loss曲线图。

### 2.8 实验结果
下面展示了部分实测结果。为方便展示，Dice的值全部乘以了100，以保持和官方相同。FPS=1*batchsize/平均每个step时间，step时间指数据经过模型并经过一次反向传播所需要的时间。GPU设备使用的是V100。

**评测结果：**
| 实验方法 | 精度：Liver 1_Dice | 精度：Liver 2_Dice | 性能（FPS） |
| :------: | :------: | :------: | :------: |
| [UNET++官方汇报](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch) | 95.80 | 65.60 | --- |
| 使用作者提供的fold_0预训练权重 | 96.55 | 71.97 | --- |
| GPU 1P bs=1 | 6.86 | 0.08 | 1.931 |
| GPU 1P bs=2 | --- | --- | 1.450 |
| GPU 8P bs=8 | 96.59 | 71.43 | 6.922 |
| GPU 8P bs=16 | 96.68 | 70.43 | 6.283 |
| NPU 1P bs=1 | 6.02 | 0.05 | 2.477 |
| NPU 1P bs=2 | --- | --- | 2.509 |
| NPU 8P bs=8 | 96.67 | 71.42 | 4.209 |
| NPU 8P bs=16 | --- | --- | 5.247 |

注：测试8P性能时，程序只会输出每张卡上的结果。因此得到的也是8P情况下单卡的性能，并且结果会输出8次。您需要将结果乘以8，以得到真实的8P性能。后续我们可能会更新这部分代码，使得8p仅输出一次结果，不需要用户手动乘以8。

备注：

1.使用作者的GPU 1P是无法收敛的，基于此开发的NPU 1P也表现出了同样的现象，可以通过loss曲线图来观察模型的训练行为。loss曲线图可以在RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/progress.png中找到，我们使用O1 level的NPU 1P和GPU 1P保持了相同的loss曲线图，若更改为O2 level，则会观察到NPU 1P出现断崖式的loss曲线上浮波动，这很可能是因为溢出了。而GPU 8P是依据UNET开发的，而不是UNET++，结果是GPU 8P和NPU 8P效果表现很好，而且和UNET++作者提供预训练过的模型的loss曲线图吻合。

2.该模型的推理过程是单幅图像推理的，即推理阶段的batchsize=1。所以在训练结束后的验证阶段，即使使用了多卡程序，也只会有一块GPU/NPU被占用。

3.请注意，我们只使用了fold 0，即第一折交叉验证实验。完整的实验应该是包含五折交叉验证的。可以使用如下的命令形式来依次开启5个独立的、顺序的交叉实验。
```
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 Task003_Liver $FOLD
done
```

4.如果训练一定epoch后被中断，可以在启动训练时，添加额外的参数--continue_training来继续上次的训练进度。

5.可以在启动训练时，添加额外的参数--validation_only，这会使模型跳过训练，直接读取RESULTS_FOLDER内的model_latest.model来进行验证集的精度测试。如果您认为训练的周期实在过长，可以在中间的某个epoch终止训练，并通过该条方法来跳过训练epoch判断，读取权重后直接进行验证推理。模型每50个epoch会存储一次权重，您可以事先通过观察loss曲线图中绿线的趋势来判断模型是否训练良好，按实际测试来看，绿线最终能达到约右侧纵轴的0.9左右处。

6.该模型的验证集测试，会对27张图像进行推理。单张图像的推理时间在10分钟-45分钟不等。在验证集上的推理时间预计为6h左右，在GPU上耗时更久。

7.不推荐将batchsize调得过大，以GPU 1P batchsize=4为例，在拥有近30G显存的V100上会爆显存。在拥有10G显存的1080ti上，batchsize只能设置为1。

8.在部分服务器上，做性能测试实验时，输出中文字符后会出现卡顿，无法阅读期望的输出文字。可以提前使用“script”命令让终端将程序的全部输出保存至文件中。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md


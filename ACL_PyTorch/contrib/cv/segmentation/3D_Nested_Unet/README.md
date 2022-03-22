# 3D_Nested_Unet模型PyTorch离线推理指导

**本教程的文件及其说明**
```
推理工具
├── benchmark.aarch64             //离线推理工具（适用ARM架构），可能需要用户自行编译获得
├── benchmark.x86_64              //离线推理工具（适用x86架构），可能需要用户自行编译获得
脚本文件
├── set_env.sh                    //NPU环境变量 
├── clear2345.sh                  //清理文件、合并结果脚本
├── get_dataset_info.py           //用于获取二进制数据集信息的脚本 
├── 3d_nested_unet_pth2onnx.py    //生成ONNX模型文件的程序
├── 3d_nested_unet_preprocess.py  //数据前处理，生成输入bin文件的程序 
├── 3d_nested_unet_postprocess.py //数据后处理，合并输出bin生成推理结果的程序
├── onnx_infer.py                 //评测GPU性能的程序
├── change_infer_path.py          //修改实验路径的程序
模型及权重文件（模型文件过大，很可能已经从本仓中移除）
├── nnunetplusplus.onnx           //ONNX模型文件
├── nnunetplusplus.om             //OM模型文件
其他文件
├── README.md                     //快速上手指导，过程内容和本文大致相同
├── new.patch                     //修改源代码的补丁
├── requirements.txt              //环境依赖，由pip freeze > re.txt生成
权重文件download_models文件夹（该文件夹可能被打包上传至别处，请用户提前下载该文件）
├── Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/*   //内含权重文件
├── Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/plans.pkl  //实验配置文件
备份文件backup文件夹（该文件夹可能被打包上传至别处，请用户提前下载该文件）
├── nnUNet_preprocessed/          //待拷贝的实验配置文件
├── output-gpu/                   //在GPU上的全部推理结果，内含GPU精度结果
├── output-npu/                   //在NPU上的全部推理结果，内含NPU精度结果
├── nnunetplusplus_prep_bin.info  //对MSD数据集Task03中第11号图像生成的info文件
├── perf_vision_batchsize_1_device_0.txt  //NPU上的性能结果
└── perf_T4gpu_batchsize_1.txt            //GPU上的性能结果
```
**关键环境：**
| 依赖名 | 版本号 |
| :------: | :------: |
| CANN  | 5.1.RC1.alpha001 |
| CANN（仅在atc转换OM时）  | 5.0.3 / 5.1.RC1.alpha001 |
| CANN（除了使用atc以外的实验步骤时）  | 5.0.3 / 5.0.4 / 5.1.RC1.alpha001 |
| python  | ==3.7.5 |
| torch   | >=1.6.0 (cpu版本即可) |
| batchgenerators  | ==0.21 |
| numpy  | 无特定版本要求 |
| pandas  | 无特定版本要求 |
| pillow  | 无特定版本要求 |
| SimpleITK  | 无特定版本要求 |
| scikit-image  | 无特定版本要求 |
| 其他依赖可在后文实验步骤中查阅  | 未指明 |

**相关链接：**
| 名称和地址 | 说明 |
| :------: | :------: |
| [UNET官方代码仓](https://github.com/MIC-DKFZ/nnUNet)  | UNET官方框架。 |
| [UNET++官方代码仓](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch)  | 依据UNET官方框架进行开发的UNET++官方代码。 |
| [MSD数据集（Medical Segmentation Decathlon）](http://medicaldecathlon.com/)  | 医学十项全能数据集，内含10个子任务，本文只对任务3肝脏任务进行验证。数据图像均为三维灰度图像，您可以下载使用ITK-SNAP工具来可视化图像。 |
| [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)  | 三维图像可视化工具。 |
| [UNET++模型权重文件](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch)  | UNET++作者提供的模型权重，在官方仓中“How to use UNet++”章节中存有链接。 |
| 权重文件download_models文件夹  | 本文所使用的，只节选了fold_0及plans.pkl的权重文件。若无链接，可下载UNET++作者提供的权重文件。 |
| 备份文件backup文件夹  | 本文所使用的，相关实验配置文件。链接位于：obs://ascend-pytorch-model-file/验收-推理/cv/segmentation/3D_Nested_Unet/实验配置文件、推理结果、性能参考文件/ |
| [benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  | 在310上进行推理所需的可执行文件。或许更新的msame工具也可以使用。 |
## 1 环境准备 

### 1.1 获取源代码
下载官方代码仓，并退回至指定版本，以保证代码稳定不变。本文以下教程与模型推理指导书保持相同。
```
cd /home/hyp/
git clone https://github.com/MrGiovanni/UNetPlusPlus.git
cd UNetPlusPlus
git reset e145ba63862982bf1099cf2ec11d5466b434ae0b --hard
```

### 1.2 安装依赖，修改模型代码  
```
cd /home/hyp/UNetPlusPlus/
patch -p1 < ../new.patch  # 载入代码修改补丁
cd pytorch
pip install -e .
pip install batchgenerators==0.21  # 该依赖十分关键，指定其版本再手动安装一次

# 您也可以通过requirements来安装依赖包，但我们不推荐该方法
pip install -r requirements.txt
```
patch命令的最后一个参数需要指定本仓中的new.patch文件的路径。由于该模型需要将命令注册到环境中才能找到正确的函数入口，所以我们仍然需要一步pip来将代码注册到环境中。除此之外，每次将代码文件进行大幅度地增减时，“pip install -e .”都是必须的，否则很可能出现“import nnunet”错误。

我们不推荐您使用requirements.txt的方式来安装环境，因为这很可能遗漏nnunet的注册步骤，使得后续实验无法进行。

注：如果在执行“pip install -e .”或在后面的实验过程中，仍然出现了环境包或模块的安装或导入错误，则很可能需要重新手动安装部分包。我们认为，原作者没有完全指明必要的依赖，而那些隐藏的依赖目前已经升级了多个版本，导致各个依赖间的关系出现变化，进而使得如今完全按作者的描述安装依赖是不可行的。我们在多个服务器上，已观测到仍然可能出现异常的包有但不仅限于：
 - torch (CPU版本即可)
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

### 1.3 准备数据集及环境设置
该模型是依赖于[UNET官方代码仓](https://github.com/MIC-DKFZ/nnUNet)而进行的二次开发，依据UNET的描述，整个实验流程大体可描述为“数据格式转换->数据预处理->训练->验证->推理”。中间不可跳过，因为每一个后续步骤都依赖于前一个步骤的结果。您可以参照官方说明进行数据集设置，但过于繁琐。下面我们将描述其中的核心步骤及注意事项，必要时通过提供中间结果文件来帮助我们跳过一些步骤。

#### 1.3.1 设置nnunet环境变量
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

#### 1.3.2 获取数据集
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

#### 1.3.3 数据格式转换
在environment文件夹内，使用nnunet的脚本命令，对解压出的Task03_Liver文件夹中的数据进行数据格式转换。该脚本将运行约5分钟，转换结果将出现在nnUNet_raw_data_base子文件夹中。
```
nnUNet_convert_decathlon_task -i Task03_Liver -p 8
```
如果您的设备性能较差或者该命令在较长时间后都未结束，您可以将参数-p的数值调小，这将消耗更多的时间。

注：若您在之后的实验过程中想要重置实验或者数据集发生严重问题（例如读取数据时遇到了EOF等读写错误），您可以将nnUNet_preprocessed、nnUNet_raw_data_base和RESULTS_FOLDER下的文件全部删除，并从本节开始复现后续过程。

#### 1.3.4 实验计划与预处理
nnunet十分依赖数据集，这一步需要提取数据集的属性，例如图像大小、体素间距等，并生成后续实验的配置文件。若删减数据集图像，都将使后续实验配置发生变化。使用nnunet的脚本命令，对nnUNet_raw_data_base中的003任务采集信息。这个过程将持续半小时至六小时不等，具体时间依赖于设备性能，转换结果将出现在nnUNet_preprocessed子文件夹中。
```
nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity
```
我们观察到，该过程很可能意外中断却不给予用户提示信息，这在系统内存较小时会随机发生，请您确保该实验过程可以正常结束。如果您的设备性能较差或者较长时间后都未能正常结束，您可以改用下面的命令来降低系统占用，而这将显著提升该步骤的运行时间。实践来看，通过输入free -m命令，如果系统显示的available Mem低于30000或在30000左右，则我们推荐您使用下面的命令。
```
nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity -tl 1 -tf 1
```
注：若在后续的实验步骤中出现形如“RuntimeError: Expected index [2, 1, 128, 128, 128] to be smaller than self [2, 3, 8, 8, 8] apart from dimension 1”的错误，请删除environment/nnUNet_preprocessed/Task003_Liver/以及environment/nnUNet_raw_data_base/nnUNet_cropped_data/下的所有文件，然后重新完成本节内容。

#### 1.3.5 拷贝实验配置文件
由于nnunet的实验计划与预处理中，对数据集的划分存在随机性，为了保证后续实验的可控性，我们提供了一些支撑材料，位于backup文件夹内。其中有一份可用的实验配置文件，即设定了训练集、验证集的划分。请将这些文件覆盖到environment中。

注：请用户自行检查：若backup/nnUNet_preprocessed/内的文件为.json格式，请将其格式手动修改为.pkl格式，保持文件名不变，之后再进行拷贝。
```
# 拷贝实验计划的.pkl文件和对数据集划分的.pkl文件至environment中
cp -rf /home/hyp/backup/nnUNet_preprocessed /home/hyp/environment/
```
在environment中创建一个新的子文件夹名为input，用于存放待推理的图像，同时再创建一个output文件夹用于存放模型的推理输出，请勿在以上两个文件夹中存放多余无关的文件。
```
cd environment
mkdir input output
```
splits_final.pkl中存储了对数据的划分，27张图片编号如下所示。我们需要将这些验证集图像（存放于nnUNet_raw_data_base/nnUNet_raw_data/Task003_Liver/imagesTr/拷贝到指定文件夹input下，作为我们的待推理图像，使用create_testset.py来完成验证集的迁移复制。当然您也可以自己指定想要推理的文件夹路径。
```
# 原始图像文件名形如liver_3_0000.nii.gz、liver_128_0000.nii.gz
# 验证集图片编号：3, 5, 11, 12, 17, 19, 24, 25, 27, 38, 40, 41, 42, 44, 51, 52, 58, 64, 70, 75, 77, 82, 101, 112, 115, 120, 128
cd /home/hyp/UNetPlusPlus/pytorch/nnunet/inference
python create_testset.py /home/hyp/environment/input/
```
注：该步骤与UNET官方教程不同，官方使用nnUNet_raw_data_base/nnUNet_raw_data/Task003_Liver/imagesTs/下的图像作为待推理图像，图像来自测试集，而本教程使用的是验证集。

#### 1.3.6 获取权重文件
该模型采用了五重交叉验证的方法，因此作者提供的预训练的权重文件也分为五个文件夹，分别代表着5个fold（交叉）的结果。实测后，各个fold的精度都相差不大，浮动大约在1%以内，鉴于计算资源的考虑，整个实验过程我们只采用fold 0（第一个交叉实验）的结果。

下载预训练过的[模型参数权重download models](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch)，在environment下创建一个新的子文件夹download_models用于存放下载得到的压缩包，将该压缩包解压后得到五个文件夹及一个配置文件：fold_0, fold_1, fold_2, fold_3, fold_4, plans.pkl。

本文教程日后可能会单独提供fold_0及plans.pkl的压缩包，如果有，用户可以自行下载使用。

将其中的fold_0文件夹和plans.pkl拷贝至environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/下，模拟我们已经完成了训练过程，请提前创建相关子文件夹。
```
cd environment
cp -rf download_models/* /home/hyp/environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/
```
最终文件结构目录如下：
```
environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/
├── fold_0/
│   ├── ...
│   ├── model_final_checkpoint.model
│   ├── model_final_checkpoint.model.pkl
│   └── ...
└── plans.pkl
```

#### 1.3.7 设置推理实验相关路径
后续推理实验通常要使用多个路径参数，使用时十分容易造成混淆。因为在前文中我们已经设置了nnunet环境变量，所以我们可以认为该模型的相关路径都是稳定的，不会经常变动。为了让后续的实验更加便捷，我们可以在程序中设置好路径作为默认参数。使用change_infer_path.py来完成这个操作，参数为三个绝对路径（以下三个fp不能指向同一个目录）。
```
python change_infer_path.py -fp1 INFERENCE_INPUT_FOLDER –fp2 INFERENCE_OUTPUT_FOLDER -fp3 INFERENCE_SHAPE_PATH
#例：python change_infer_path.py -fp1 /home/hyp/environment/input/ -fp2 /home/hyp/environment/output/ -fp3 /home/hyp/environment/

```
以上的三个路径参数的具体说明如下，我们推荐将这些路径指向environment文件夹内，便于用户检索：
 - INFERENCE_INPUT_FOLDER：存放待推理图像的文件夹。（该文件夹在1.3.5节中被创建）
 - INFERENCE_OUTPUT_FOLDER：推理完成后，存放推理结果的文件夹。（该文件夹在1.3.5节中被创建）
 - INFERENCE_SHAPE_PATH：存放文件all_shape.txt的目录。在后续实验过程中会被介绍到，在该目录下会生成一个all_shape.txt，存储着当前待推理图像的属性。这是一个中间结果文件，用户无需具体了解。
 
最后，您可以打开项目代码中的UNetPlusPlus/pytorch/nnunet/inference/infer_path.py查看修改的结果，修改后的效果示例如下所示：
```
# 以下两项为固定值，为历史需求变更后的版本遗留项，请保证为None
INFERENCE_BIN_INPUT_FOLDER = None
INFERENCE_BIN_OUTPUT_FOLDER = None

# 以下三项为change_infer_path.py修改后的三个路径
INFERENCE_INPUT_FOLDER = '/home/hyp/environment/input/'
INFERENCE_OUTPUT_FOLDER = '/home/hyp/environment/output/'
INFERENCE_SHAPE_PATH = '/home/hyp/environment/'
```
注：在本节中，您可能会首次打开并查看项目代码。如果文件中存在中文字符，在您的软件上可能会显示乱码，请更换编码方式为UTF-8来查看。

#### 1.3.8 拷贝实验结果
推理需要对验证集中未经训练的27张图像进行推理，实测上在NPU上完成全部的推理需要2-4天时间。由于推理过程过于繁琐，我们额外提供了一份含有在fold 0设置下的全部推理结果的附加文件，也包含在NPU上的完整推理流程下的推理结果。后文将以编号11的图像为例，讲解如何进行单幅图像的推理，而其他编号的图像也可以遵循同样的方法来得到，进而复现出所有的推理结果。所有验证集图像的编号如下，将backup/output-npu/中的NPU推理结果拷贝至INFERENCE_OUTPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/output/）下。
```
# 结果图像文件名形如liver_5.nii.gz、liver_112.nii.gz
# 图片编号同1.3.5节中所介绍的：3, 5, 11, 12, 17, 19, 24, 25, 27, 38, 40, 41, 42, 44, 51, 52, 58, 64, 70, 75, 77, 82, 101, 112, 115, 120, 128
cp -rf /home/hyp/backup/output-npu/* /home/hyp/environment/output/
```
注：output-npu和output-gpu下的summary.json即为整个实验在NPU和GPU上的精度评测结果，仅供参考。在2.9节中我们会替换掉它生成新的评测结果。若用户发现存在plans.json文件，请将其后缀格式修改为.pkl。

### 1.4 获取[benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer) 
将编译好的benchmark.x86_64或benchmark.aarch64放到当前工作目录。您可以使用如下命令来确认自己的系统是x86架构还是aarch架构。
```
uname -a
```

## 2 离线推理 

### 2.1 生成om模型
下面简要介绍了离线推理中的重要步骤所使用的程序，它们都必须接受一个用户提供的路径参数--file_path：
 - 3d_nested_unet_pth2onnx.py：转换模式。加载预训练的模型并转化为onnx模型，输出的onnx文件为--file_path。
 - 3d_nested_unet_preprocess.py：拆分模式。数据前处理，将INFERENCE_INPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/input/）下的待推理图像切割子图，生成一批输入.bin文件，存放到--file_path下。
 - 3d_nested_unet_postprocess.py：组合模式。数据后处理，将--file_path下的输出.bin文件合并出推理结果，推理结果会存放到INFERENCE_OUTPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/output/）下。

首先让模型载入预训练好的权重，将其转化为onnx模型，输出文件为一个指定路径下的nnunetplusplus.onnx，暂且将其置于environment内。
```
python 3d_nested_unet_pth2onnx.py --file_path /home/hyp/environment/nnunetplusplus.onnx
```
注：首次运行该程序时，将消耗比平时更多的时间。

之后我们需要将onnx转化为om模型，先使用npu-smi info查看设备状态，确保device空闲后，执行以下命令。这将生成batch_size为1的om模型，其输入onnx文件为nnunetplusplus.onnx，输出om文件命名为nnunetplusplus，这将在当前路径下生成nnunetplusplus.om文件，后面的--input_format和--input_shape参数则指代了该模型的输入图像规格与尺寸。
```
cd environment
atc --framework=5 --model=nnunetplusplus.onnx --output=nnunetplusplus --input_format=NCDHW --input_shape="image:1,1,128,128,128" --log=debug --soc_version=Ascend310
```
注：我们注意到在CANN 5.0.3上，atc命令可以通过，但在CANN 5.0.4上却会报错：RuntimeError: ({'errCode': 'E90003', 'detailed_cause': 'tuple_reduce_sum not support'}, 'Compile operator failed, cause: Template constraint, detailed information: tuple_reduce_sum not support.')。最终本文在CANN 5.1.RC1.alpha001下又得以通过。

### 2.2 删除指定的待推理图像的结果文件
本质上，若在INFERENCE_INPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/input/）中存在输入图像而在INFERENCE_OUTPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/output/）中不存在结果图像，二者的差集便是模型需要进行推理的内容，接着模型便是随机挑选一张未经推理的图像进行推理，这个随机性是由多个进程的IO读取速率来决定的。

因此，我们将INFERENCE_OUTPUT_FOLDER中的某个指定编号的文件删除掉，就可以对该图像进行一次推理流程了。删除输出结果文件夹INFERENCE_OUTPUT_FOLDER中的编号为11的结果，模拟已经完成了其余26张图像的推理，并准备开始对编号11的图像进行推理。
```
# 全部验证集图像的编号：3, 5, 11, 12, 17, 19, 24, 25, 27, 38, 40, 41, 42, 44, 51, 52, 58, 64, 70, 75, 77, 82, 101, 112, 115, 120, 128
rm /home/hyp/environment/output/liver_11.nii.gz
```
如果您想推理其他图像，删除在INFERENCE_OUTPUT_FOLDER中的其他编号的结果文件，使得与INFERENCE_INPUT_FOLDER的差集不为空集即可。我们推荐您每次只推理一张图像，否则您无法确切知道模型目前正在推理哪张图像，以及当前推理的进度。如果差集较大，则很可能占据超过预期的存储空间。

### 2.3 数据预处理后切割子图，生成待输入bin文件
遵从UNET的实验流程，一张待推理的图像会被切割出1000至4000张的子图，我们需要将这些子图存储为.bin文件，存放在指定目录下，暂且先定为environment/input_bins。使用3d_nested_unet_preprocess.py，参数--file_path指定为想要生成输入bin文件的目录，请用户自行创建该文件夹。
```
python 3d_nested_unet_preprocess.py --file_path /home/hyp/environment/input_bins/
```
该程序执行成功后，会在--file_path下生成大量的.bin文件，并且在INFERENCE_SHAPE_PATH（在1.3.7节中被设置为/home/hyp/environment/）下生成一个all_shape.txt文件，该文件存储了当前待输入图像的部分属性信息，这些信息将在后续的实验过程中帮助输出.bin的结果合并，使用过程中无需查阅里面的内容。

注：请确保有充足的硬盘空间。若使用310设备，遵从UNET的实验流程设计，推理一副图像，预计消耗200GB至800GB（多为300GB左右，上限受原始图像尺寸影响，800GB是一个预估值）的额外存储空间，耗时半小时至两小时不等。待推理的图像共有27张，不可能一次性将所有图像都推理完毕，因此我们只能采用逐个图像推理，之后立即做结果合并，然后删除掉使用过的bin文件，重复此过程。

### 2.4 生成info文件
使用UNetPlusPlus/pytorch/nnunet/inference/gen_dataset_info.py，读取INFERENCE_INPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/input/）中全部文件的路径，即生成的预处理数据.bin的路径，进而生成对应的info文件，作为benchmark工具推理的输入，将结果命名为nnunetplusplus.info，两个参数128指代了模型的输入尺寸。
```
python gen_dataset_info.py bin ./environment/input_bins nnunetplusplus_prep_bin.info 128 128
```
这个操作同时会在nnunetplusplus_prep_bin.info所在目录下额外生成四个子文件：sth1.info, sth2.info, sth3.info, sth4.info。它们是对nnunetplusplus_prep_bin.info的不重叠的有序拆分，有了这些拆分的info文件，便于我们同步使用4个310设备进行推理，加快实验进度。

### 2.5 使用benchmark工具进行推理
确保device空闲，将benchmark工具与上节生成的.info文件放于同一目录下，使用benchmark工具同步开启一个或四个进程进行推理。参数-device_id指代了使用的设备编号，-om_path指代了使用的om模型，-input_text_path指代了采用的info文件，-output_binary=True指代了将结果保存为.bin。
```
source set_env.sh  # 激活NPU环境
# 方法一：使用总的nnunetplusplus_prep_bin.info，使用1个310进行推理
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./environment/nnunetplusplus.om -input_text_path=nnunetplusplus_prep_bin.info -input_width=128 -input_height=128 -output_binary=True -useDvpp=False

# 方法二：使用拆分的四个info，使用4个310进行推理，全部推理结束后必须使用clear2345.sh脚本。可以通过打开四个session来完成
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./environment/nnunetplusplus.om -input_text_path=sth1.info -input_width=128 -input_height=128 -output_binary=True -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./environment/nnunetplusplus.om -input_text_path=sth2.info -input_width=128 -input_height=128 -output_binary=True -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=2 -batch_size=1 -om_path=./environment/nnunetplusplus.om -input_text_path=sth3.info -input_width=128 -input_height=128 -output_binary=True -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=1 -om_path=./environment/nnunetplusplus.om -input_text_path=sth4.info -input_width=128 -input_height=128 -output_binary=True -useDvpp=False
```
这会在当前路径下，自动生成result文件夹，里面有形如dumpOutput_device0的子文件夹，存放着对info文件中记录的每个输入.bin的推理输出.bin，“device0”指在第0个设备上的运行结果。而生成的形如perf_vision_batchsize_1_device_0.txt的文件，则记录了310推理过程中的部分指标与性能。

注：本节内容将会产生大量的输出.bin文件，请使用df -h及时观测硬盘剩余空间。如果实验进行到一半，硬盘空间紧张，请查阅下节内容。

### 2.6 清除多余的结果
上节中使用的benchmark工具，对每张输入.bin会输出五个输出结果.bin文件，而只有其中之一是我们所需要的结果。我们需要修改程序脚本中的路径。找到项目UNetPlusPlus下的clear2345.sh，该脚本用于删除310输出结果中后缀带有2、3、4、5的冗余.bin文件（保留后缀带有1的.bin文件），并将所有的.bin文件都移动到同一个文件夹下（例如放置于device0卡的输出路径），便于后续的结果合并搜索指定子图结果。我们将该脚本中的rm命令参数替换为正确的310输出路径。之后的mv命令，用于将4卡的输出结果全部移动到1卡上，也要保持正确。该脚本在推理时才会用到，一份可用的示例如下：
```
# 删除多余的输出.bin文件
rm -rf ./result/dumpOutput_device*/*_2.bin
rm -rf ./result/dumpOutput_device*/*_3.bin
rm -rf ./result/dumpOutput_device*/*_4.bin
rm -rf ./result/dumpOutput_device*/*_5.bin

# 将其他文件夹的.bin结果移动到同一个目录下
mv ./result/dumpOutput_device1/* ./result/dumpOutput_device0/
mv ./result/dumpOutput_device2/* ./result/dumpOutput_device0/
mv ./result/dumpOutput_device3/* ./result/dumpOutput_device0/
```
通常来说，您只需要对上述脚本设置一次即可。执行该脚本将多余的.bin文件删除。当所有设备都推理结束后，也请执行一次该脚本，确保所有结果都位于同一文件夹下。
```
bash clear2345.sh
```
注：clear2345.sh脚本可与前一节同步使用。及时使用df -h命令查看硬盘剩余空间，适时调用该脚本清理多余的后缀为2、3、4、5的输出.bin文件，使得该实验仍可以在存储空间较小的设备上运行。以4卡并行为例，每半小时运行一次该脚本，可以清理出约150GB-200GB的存储空间。在前一节内容全部完成后，也要调用一次该脚本，将4卡的结果都移动到dumpOutput_device0文件夹中，保证dumpOutput_device0文件夹中保留有全部的输出.bin文件。

### 2.7 将结果.bin文件合并为最终推理结果
使用3d_nested_unet_postprocess.py，参数--file_path指定为经310推理生成的.bin文件的目录，也就是将result/dumpOutput_device0/下的.bin文件做结果合并。生成的推理结果会输出到INFERENCE_OUTPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/output/）下。
```
python 3d_nested_unet_postprocess.py --file_path /home/hyp/result/dumpOutput_device0/
```

### 2.8 重复实验
截止目前，我们已经完成了1张编号为11的待推理图像的推理结果，删除benchmark工具生成的相关文件，即result/dumpOutput_device*/，释放硬盘空间。

若用户希望复现其他结果，请重复2.2至2.8的步骤，直至全部的验证集图片都推理完毕。

### 2.9 精度评测
推理完成后，我们需要对全部结果做精度验证。将INFERENCE_OUTPUT_FOLDER（在1.3.7节中被设置为/home/hyp/environment/output/）下的结果拷贝至environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/下，模拟我们已经使用模型完成了训练过程，并且进入了验证阶段。请用户自行创建相关子文件夹。
```
cp -rf /home/hyp/environment/output/* environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/
```
请确保有27个结果图像已经位于上述的validation_raw文件夹中。然后使用nnUNet_train脚本命令开始评测精度，--validation_only表明我们不需要重新训练，直接进入验证步骤。
```
nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 003 0 --validation_only
```
注：首次运行nnUNet_train命令时，模型将开始对数据集解包，这将消耗比平时更多的时间。

实验的精度将记录在environment/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json中，您可以参照如下的结构树来找到其中的Dice指标。结果存在浮动是正常现象。
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
│       │   ├── "Dice": 0.9655123016429166
│       │   └── ...
│       └── "2"
│           ├── ...
│           ├── "Dice": 0.719350267858144
│           └── ...
├── "task": "Task003_Liver"
└── "timestamp"
```
这是在第一折交叉验证下的结果，验证集图像只有27张，本文的肝脏数据是在不同的实验仪器下采集的，图像尺寸与图像质量均存在较大差异。选用不同的交叉必然会导致不同的实验结果，但对精度达标的目标来说影响不大。

### 2.10 性能评测
GPU上的性能使用onnx_infer.py来计算，需要在T4服务器上执行。您也可以在从backup/perf_T4gpu_batchsize_1.txt中直接查看性能结果。
```
python onnx_infer.py nnunetplusplus.onnx 1,1,128,128,128
```
NPU上的性能使用benchmark工具来计算，需要在310服务器上执行。使用benchmark前需要激活set_env.sh环境变量。您也可以在前面benchmark的输出文件夹result/下找到perf_vision_batchsize_1_device_0.txt文件，该文件由benchmark默认生成，在backup中我们也提供了一份实测样本，该结果与以下命令得到的结果几乎相同。
```
source set_env.sh
./benchmark.x86_64 -round=20 -om_path=nnunetplusplus.om -device_id=0 -batch_size=1
```
以下是实测结果，可供参考：
```
NPU 310性能：ave_throughputRate = 0.235349samples/s, ave_latency = 4249.14ms
GPU T4性能：Average time spent: 2.68s
```

**评测结果：**   
| 模型      | 官网pth精度  | GPU推理精度 | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------: | :------:  | :------:  | 
| 3D nested_unet bs1  | [Liver 1_Dice (val):95.80, Liver 2_Dice (val):65.60](https://github.com/MrGiovanni/UNetPlusPlus/tree/master/pytorch) | Liver 1_Dice (val):96.55, Liver 2_Dice (val):71.94 | Liver 1_Dice (val):96.55, Liver 2_Dice (val):71.97 |  0.3731fps | 0.9414fps | 

备注：

1.该模型的推理过程从设计之初便不支持batchsize 2及以上，本教程全程使用了batchsize 1。

2.本应使用测试集进行精度验证的。但该数据集的测试集不支持单任务的精度测试，其测试集label是不公开的。因此本文只能使用数据集的验证集进行精度测试，这也导致了本文的一些实验步骤与官方不同。

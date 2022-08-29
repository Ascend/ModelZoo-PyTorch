# WDSR模型PyTorch离线推理指导
## 1 环境准备
1. 安装必要的依赖
```
pip3.7 install -r requirements.txt
```

2. 获取开源模型代码
```
git clone https://github.com/ychfan/wdsr.git -b master 
cd wdsr
git reset --hard b78256293c435ef34e8eab3098484777c0ca0e10
cd ..
```

3. 获取权重文件
```
wget https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip
unzip wdsr_x2.zip
rm wdsr_x2.zip
```

4. 获取数据集

[DIV2K数据集网址](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

下载`Validation Data (HR images)`和`Validation Data Track 1 bicubic downscaling x2 (LR images)`两个压缩包，新建data/DIV2K文件夹，将两个压缩包解压至该文件夹中。

5. 获取[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 准备数据集

1.获取原始数据集

    本模型支持div2k 100张图片的验证集Validation Data (HR images)和Validation Data Track 1 bicubic downscaling x2 (LR images)两个数据集，上传数据集到服务器任意目录。在源码包中新建data/DIV2K文件夹，将两个压缩包解压至该文件夹中

2.数据预处理。

    执行预处理脚本，生成数据集预处理后的bin文件
    ```
    python3.7 Wdsr_prePorcess.py --lr_path ./data/DIV2K/DIV2K_valid_LR_bicubic/X2/ --hr_path ./data/DIV2K /DIV2K_valid_HR/ --save_lr_path ./DIV2K_valid_LR_bicubic_bin/X2/  --width 1020 --height 1020 --scale 2 
    ```
    第一个参数为低分辨率数据集相对路径，第二个参数为高分辨率数据集的相对路径，第三个参数为生成数据集文件的保存路径。运行成功后，在当前目录生成DIV2K_valid_LR_bicubic_bin/X2/数据集。

3.生成数据集info文件

    使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。执行gen_info脚本，输入已经获得的图片文件，输出生成图片数据集的info文件
    ```
    python3.7 get_info.py bin ./DIV2K_valid_LR_bicubic_bin/X2/ wdsr_bin.info 1020 1020 
    ```
    第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件的保存的路径。运行成功后，在当前目录中生成wdsr_bin.info。

## 3 离线推理

310和310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

1.模型转换

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    a.获取权重文件。

        从源码包中获取权重文件epoch_30.pth。

    b.导出onnx文件。

        执行pth2onnx.py脚本，获得onnx文件。
        ```
        python3.7 Wdsr_pth2onnx.py --ckpt epoch_30.pth --model wdsr --output_name wdsr.onnx --scale 2
        ```
        获得wdsr.onnx文件。第一个参数是pth权重文件所处的位置。第二个参数是要导入的模型名称，第三个参数是输出onnx的名称。第四个参数是缩放大小。
    c.使用ATC工具将ONNX模型转OM模型。

        i.设置 atc 工作所需要的环境变量。
            ```
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
        ii.使用atc工具将onnx模型转换为om模型。获得wdsr_bs1.om文件。
            ``` 
            atc --framework=5 --model=wdsr.onnx --output=wdsr_bs1 --input_format=NCHW --input_shape="image:1,3,1020,1020" --log=debug --soc_version=Ascend${chip_name}
            ```
            ${chip_name}可通过 npu-smi info 指令查看。

2.开始推理验证。

    a.使用Benchmark工具进行推理。

        i.增加benchmark.{arch}可执行权限。
            ```
            chmod u+x benchmark.x86_64
            ```
        ii.推理。
            ```
            ./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=1 -om_path=./wdsr_bs1.om -input_text_path=./wdsr_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False
            ```
    b.精度验证。

        调用Wdsr_postProcess.py脚本。
        ```
        python3.7 Wdsr_postProcess.py --bin_data_path ./result/dumpOutput_device3/ --dataset_path ./data/DIV2K/DIV2K_valid_HR/ --result result_bs1.txt --scale 2
        ```
        第一个参数为生成推理结果所在路径，第二个参数为高分辨率图片所在位置，第三个参数为生成结果文件，第四个参数为缩放图片大小。

**评测结果：**

| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    | 310P性能    |
| :------: | :------: | :------: | :------:  | :------:  | :------:  |
| WDSR bs1  | psnr:34.76 | psnr:34.7537 | 6.8778fps | 5.936fps | 11.0603fps |
| WDSR bs8  | psnr:34.76 | psnr:34.7537 | 6.5681fps | 6.124fps | 8.9136fps |

**备注：**

- 在om的bs16版本在310设备上运行离线推理会out of memory。
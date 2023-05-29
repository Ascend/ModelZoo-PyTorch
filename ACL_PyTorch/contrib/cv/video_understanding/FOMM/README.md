# FOMM模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

FOMM模型最早是Aliaksandr Siarohin等人在发表的《First Order Motion Model for Image Animation》一文中提到的用于图像动画化（image animation）的模型。图像动画化任务是指给定一张原图片和一个驱动视频，通过视频合成，生成主角为原图片，而动画效果和驱动视频一样的视频。以往的视频合成往往依赖于预训练模型来提取特定于对象的表示，而这些预训练模型是使用昂贵的真实数据注释构建的，并且通常不适用于任意对象类别。而FOMM的提出很好的解决了这个问题。


- 参考实现：

  ```
  url=https://github.com/AliaksandrSiarohin/first-order-model.git
  commit_id=3d152de07e51dcd00358475c0defbf8f85b2ab3e
  ```




## 输入输出数据<a name="section540883920406"></a>
- kp detector
   - 输入数据

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | RGB_FP32 | 1 x 3 x 256 x 256 | NCHW         |


   - 输出数据

      | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
      | -------- | -------- | -------- | ------------ |
      | value    | FLOAT32  | 1 x 10 x 2     | ND           |
      | jacobian | FLOAT32  | 1 x 10 x 2 x 2 | NCHW         |

- generator
   - 输入数据

      | 输入数据    | 数据类型 | 大小                              | 数据排布格式      |
      | ----------- | -------- | --------------------------------- | ----------------- |
      | source_imgs | RGB_FP32 | 1 x 3 x 256 x 256                 | NCHW              |
      | kp_driving  | RGB_FP32 | 1 x 10 x 2；<br/>1 x 10 x 2 x 2； | NCHW              |
      | kp_source   | RGB_FP32 | 1 x 10 x 2；<br/>1 x 10 x 2 x 2； | NCH；<br />NCHW； |


   - 输出数据

      | 输出数据 | 数据类型 | 大小                                                         | 数据排布格式 |
      | -------- | -------- | ------------------------------------------------------------ | ------------ |
      | out      | FLOAT32  | 1 x 11x 64 x 64；<br />1 x 11 x 3 x 64 x 64；<br />1 x 1 x 64 x 64；<br />1 x 3 x 256 x 256；<br />1 x 3 x 256 x 256； | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/AliaksandrSiarohin/first-order-model.git
   cd first-order-model
   git reset 3d152de07e51dcd00358475c0defbf8f85b2ab3e --hard
   mv ../fomm.patch ./
   git apply fomm.patch
   cd ..
   export PYTHONPATH=./first-order-model:$PYTHONPATH
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 获取其他源码。

   获取[pose_model.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FOMM/PTH/pose_model.pth)，保存到主目录下。

   ```shell
   git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
   cd maskrcnn-benchmark
   python setup.py install
   cd ..
   git clone --recursive https://github.com/AliaksandrSiarohin/pose-evaluation
   mkdir pose-evaluation/pose_estimation/network/weight/
   mv pose_model.pth pose-evaluation/pose_estimation/network/weight/
   cd pose-evaluation
   mv ../pose1.patch ./
   git apply pose1.patch
   cd pose_estimation
   mv ../../pose2.patch ./
   git apply pose2.patch
   cd ../../
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持taichi验证集。下载方式参考[开源仓](https://github.com/AliaksandrSiarohin/first-order-model/tree/master/data/taichi-loading)。

   上传数据集到源码包路径下。目录结构如下：

   ```
   data
      |-- taichi
      `-- |-- test
          |   `-- 0Q914by5A98#010440#010764.mp4
          |   `-- 8hLvlQrXI6U#007700#007984.mp4
          |   `-- 8hLvlQrXI6U#008247#008392.mp4
          |   `-- ...
          `-- train
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行FOMM_preprocess.py脚本，完成预处理。
   ```shell
   python3 FOMM_preprocess.py --config first-order-model/config/taichi-256.yaml --data_type npy --out_dir pre_data/
   ```
   - 参数说明：
      - config：配置文件路径。
      - data_type：输出数据类型。
      - out_dir：预处理输出数据存储路径。
   运行成功后在主目录下生成pre_data文件夹。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```shell
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FOMM/PTH/taichi-cpk.pth.tar
      mkdir checkpoint
      mv taichi-cpk.pth.tar ./checkpoint/
      ```

   2. 导出onnx文件。

      1. 使用FOMM_pth2onnx.py导出onnx文件。

         运行FOMM_pth2onnx.py脚本。

         ```shell
         python3 FOMM_pth2onnx.py --config first-order-model/config/taichi-256.yaml --checkpoint checkpoint/taichi-cpk.pth.tar --outdir ./ --genname taichi-gen-bs1 --kpname taichi-kp-bs1
         ```
         - 参数说明：
            - config：配置文件路径。
            - checkpoint：权重文件路径。
            - outdir：模型输出路径。
            - genname：onnx模型名称。

         获得taichi-gen-bs1.onnx，taichi-kp-bs1.onnx文件。

      2. 优化ONNX文件。
         请访问[auto-optimizer工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         ```
         python3 modify_onnx.py --input_name taichi-gen-bs1.onnx --output_name taichi-gen-bs1_new.onnx
         ```

         获得taichi-gen-bs1_new.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=taichi-gen-bs1_new.onnx \
            --output=taichi-gen-bs1 --input_format=NCHW \
            --input_shape="source_imgs:1,3,256,256;kp_driving_value:1,10,2;kp_driving_jac:1,10,2,2;kp_source_value:1,10,2;kp_source_jac:1,10,2,2" \
            --log=error --soc_version=Ascend${chip_name} \
            --buffer_optimize=off_optimize
         atc --framework=5 --model=taichi-kp-bs1.onnx \
            --output=taichi-kp-bs1 --input_format=NCHW \
            --input_shape="input:1,3,256,256" --log=error \
            --soc_version=Ascend${chip_name} --buffer_optimize=off_optimize
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --buffer_optimizer：是否开启数据缓存优化

           运行成功后生成taichi-gen-bs1.om，taichi-kp-bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
      mkdir -p infer_out/out/
      python3 -m ais_bench --model taichi-kp-bs1.om --input pre_data/driving/ --output infer_out/ --outfmt NPY --output_dirname kpd
      python3 -m ais_bench --model taichi-kp-bs1.om --input pre_data/source/ --output infer_out/ --outfmt NPY --output_dirname kps
      python3 apart_kp_out.py --type npy --data_root infer_out --driving_dir kpd --source_dir kps
      python3 -m ais_bench --model taichi-gen-bs1.om --input pre_data/source/,infer_out/kpdv/,infer_out/kpdj/,infer_out/kpsv/,infer_out/kpsj/ --output infer_out/ --outfmt NPY --output_dirname out/
      ```

      - 参数说明：
         - ais_bench
           - model：om模型路径。
           - input：输入数据路径。
           - output：输出数据路径。
           - output_dirname：输出数据子目录。
           - output_format：输出数据类型。
         - apart_kp_out.py
           - type：数据类型。
           - data_root：推理数据路径。
           - driving_dir：driving数据的推理结果子目录。
           - source_dir：source数据的推理结果子目录。

        推理后的输出默认在当前目录infer_out下。


   3. 精度验证。

      运行下列命令验证模型精度。

      ```
      python3 FOMM_reconstruction.py --config first-order-model/config/taichi-256.yaml --png_dir checkpoint/reconstruction/png
      cd pose-evaluation/
      python3 extract.py --in_folder ../data/taichi/test/ --out_file pose_gt.pkl --is_video --type body_pose --image_shape 256,256
      python3 extract.py --in_folder ../checkpoint/reconstruction/png --out_file pose_gen.pkl --is_video --type body_pose --image_shape 256,256
      python3 cmp_with_missing.py pose_gt.pkl pose_gen.pkl
      ```

      - 参数说明：
         - FOMM_reconstruction.py
            - config：配置文件路径。
            - png_dir：图片信息保存路径。
            - data_dir：推理输出的数据保存的目录，对应前面推理命令中的--output参数，默认为infer_out。
            - pre_data：预处理后数据保存的目录，对应前面预处理命令中的--out_dir参数，默认为pre_data。
         - extract.py
            - in_folder：输入的视频或图片的保存目录。
            - out_file：输出的.pkl文件的文件名。
            - type：使用的函数的类型。
            - image_shape：帧图片的shape。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20
        ```

      - 参数说明：
        - --model：om模型路径。
        - --loop：推理次数。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| ------- | ------------ | ---------- | ---------- | --------------- |
|  310P   | 1 | taichi | ADK：6.8；<br />MKR：0.036 | kp detector：957.81<br />generator：7.75 |

该模型只支持batch size 1
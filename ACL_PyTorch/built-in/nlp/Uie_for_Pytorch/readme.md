# Uie 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。为了方便大家使用UIE的强大能力，PaddleNLP借鉴该论文的方法，基于ERNIE 3.0知识增强预训练模型，训练并开源了首个中文通用信息抽取模型UIE。该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。本任务旨在移植UIE框架至NPU侧推理。
- 参考实现：

  ```
  url=https://gitee.com/paddlepaddle/PaddleNLP.git
  tag=v2.4.7
  code_path=${PaddleNLP}/model_zoo/uie
  model_name=uie
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int64 | batchsize x max_seq_len | ND         |
  | token_type_ids    | int64 | batchsize x max_seq_len | ND         |
  | position_ids | int64 | batchsize x max_seq_len | ND |
  | attention_mask | int64 | batchsize x max_seq_len | ND |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | start_prob  | float32  | batchsize x 1 | ND |
  | end_prob | float32 | batchsize x 1 | ND          |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |                                                        |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 拉取paddleNlp代码
   ```
   git clone https://gitee.com/paddlepaddle/PaddleNLP.git
   # 切换到v2.4.7版本
   git checkout v2.4.7
   ```
   将代码拷贝到paddleNLP对应目录中
   ```
   # 默认当前目录为ACL_PyTorch/built-in/nlp/Uie_for_Pytorch
   cp *.py ${PaddleNLP}/model_zoo/uie/
   # 切换到paddleNLP uie目录
   cd ${PaddleNLP}/model_zoo/uie/
   cp infer_npu.py ./deploy/python/
   cp uie_npu_predictor.py ./deploy/python/
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   根据源码训练脚本训练模型后导出onnx模型，再使用atc工具导出om离线模型。

   1. 训练权重后导出onnx模型。

      下载标记好的[数据集](https://gitee.com/link?target=https%3A%2F%2Fbj.bcebos.com%2Fpaddlenlp%2Fdatasets%2Fuie%2Fdoccano_ext.json)放入${paddleNLP}/model_zoo/uie/data 
      ```
      # 执行以下脚本进行数据转换，执行后会在./data目录下生成训练/验证/测试集文件
      python doccano.py \
      --doccano_file ./data/doccano_ext.json \
      --task_type ext \
      --save_dir ./data \
      --splits 0.8 0.2 0 \
      --schema_lang ch
      ```
      执行以下训练脚本得到finetune之后的权重文件（推荐使用GPU环境）
       ```
       export finetuned_model=./checkpoint/model_best

      python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
         --device gpu \
         --logging_steps 10 \
         --save_steps 100 \
         --eval_steps 100 \
         --seed 42 \
         --model_name_or_path uie-base \
         --output_dir $finetuned_model \
         --train_path data/train.txt \
         --dev_path data/dev.txt  \
         --max_seq_length 512  \
         --per_device_eval_batch_size 16 \
         --per_device_train_batch_size  16 \
         --num_train_epochs 100 \
         --learning_rate 1e-5 \
         --do_train \
         --do_eval \
         --do_export \
         --export_model_dir $finetuned_model \
         --label_names 'start_positions' 'end_positions' \
         --overwrite_output_dir \
         --disable_tqdm True \
         --metric_for_best_model eval_f1 \
         --load_best_model_at_end  True \
         --save_total_limit 1 \
       ```
       可配置参数说明：

      * `model_name_or_path`：必须，进行 few shot 训练使用的预训练模型。可选择的有 "uie-base"、 "uie-medium", "uie-mini", "uie-micro", "uie-nano", "uie-m-base", "uie-m-large"。
      * `multilingual`：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型也是多语言模型，需要设置为 True；默认为 False。
      * `output_dir`：必须，模型训练或压缩后保存的模型目录；默认为 `None` 。
      * `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。
      * `per_device_train_batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
      * `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
      * `learning_rate`：训练最大学习率，UIE 推荐设置为 1e-5；默认值为3e-5。
      * `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
      * `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认100。
      * `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
      * `seed`：全局随机种子，默认为 42。
      * `weight_decay`：除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
      * `do_train`:是否进行微调训练，设置该参数表示进行微调训练，默认不设置。
      * `do_eval`:是否进行评估，设置该参数表示进行评估。

      根据当前设备环境情况，选择执行以下推理脚本得到对应onnx模型（${finetuned_model}/model.onnx）
      ```
      # cpu
      python deploy/python/infer_cpu.py --model_path_prefix ${finetuned_model}/model
      # gpu
      python deploy/python/infer_gpu.py --model_path_prefix ${finetuned_model}/model --device_id 0
      ```
      可配置参数说明：

      - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/model.pdiparams`，则传入`./export/model`。
      - `position_prob`：模型对于span的起始位置/终止位置的结果概率 0~1 之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
      - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
      - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为 4。
      - `multilingual`：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型是多语言模型，需要设置为 True；默认为 False。
      - `device_id`: GPU 设备 ID，默认为 0。


       

   2. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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
          atc --model=model.onnx \
               --framework=5 \
               --output=uie_bs${batch_size} \
               --input_format=ND \
               --input_shape="token_type_ids:${batch_size},512; \
                              input_ids:${batch_size},512; \
                              position_ids:${batch_size},512; \
                              attention_mask:${batch_size},512"\

               --log=error \
               --soc_version=Ascend${chip_name} \
               --optypelist_for_implmode="Gelu,Tanh" \
               --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`uie_bs${batch_size}.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      1）示例推理（暂只支持单batch示例，多batch请参考数据集测试）
        ```
        python3 infer_npu.py  --model_path uie_bs1.om --device_id 0 --batch_size 1
        ```

        -   参数说明：

             -   model_path：om模型
             -   device_id：芯片ID
             -   batch_size：模型bs

         执行改示例推理脚本后，终端会回显对文本进行的实体抽取和关系抽取（示例为法律场景，可自行修改构造其他场景）

      2） 数据集评估
      
      执行以下脚本测试数据集
      ```
      python evaluate_om.py \
      --om_path uie_bs${batch_size}.om \
      --test_path ./data/dev.txt \
      --batch_size ${batch_size} \
      --device_id 0 \
      --debug
      ```
      -   参数说明：

             -   om_path：om模型路径
             -   device_id：芯片ID
             -   batch_size：模型bs
             -   test_path: 测试数据集路径
             -   debug: 开启显示各个class评估数据，关闭debug只显示总的评估数据

   3. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 精度（Acc） | 性能（fps） |
| :------: | :--------:  | :--: | :--: |
|    310P3      |     1       |      100%    |  156.61    |
|    310P3      |     4       |      100%    |  162.92    |
|    310P3      |     8       |      100%    |    169.20  |
|    310P3      |     16       |    100%    |  172.80    |
|    310P3      |     32       |   100%    |  173.97    |
|    310P3      |     64       |      100%    |  172.48    |

测试tensorrt推理引擎，竞品性能参考下列数据。
| 芯片型号 | Batch Size  | 性能（fps） |
| :------: | :--------:  |  :--: |
|    T4      |     1           |  210.32    |
|    T4      |     4          |  208.46    |
|    T4      |     8           |    210.97  | 
|    T4      |     16           |  227.98    |
|    T4      |     32          |  229.67    |
|    T4      |     64           |  238.51    |
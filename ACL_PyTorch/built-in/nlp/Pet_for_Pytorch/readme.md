# Pet 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

自然语言处理任务可以通过给预训练模型提供“任务描述”等方式来进行无监督学习，但效果一般低于有监督训练。而 Pattern-Exploiting Training (PET) 是一种半监督方法，通过将输入转换为完形填空形式的短语来帮助语言模型理解任务。然后用这些短语来给无标注数据打软标签。最后在得到的标注数据集上用有监督方法进行训练。在小样本设置下，PET 在部分任务上远超有监督学习和强半监督学习方法。以 PET 为代表的提示学习与微调学习的区别如下图所示，包括数据预处理模块 Template 和标签词映射模块 Verbalizer。本任务旨在移植PET提示学习训练的Ernie模型至NPU侧推理。

- 参考实现：

  ```
  url=https://gitee.com/paddlepaddle/PaddleNLP.git
  tag=v2.4.7
  code_path=${PaddleNLP}/model_zoo/Pet
  model_name=Pet
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int64 | batchsize x max_seq_len | ND         |
  | token_type_ids    | int64 | batchsize x max_seq_len | ND         |
  | position_ids | int64 | batchsize x max_seq_len | ND |
  | attention_mask | float32 | batchsize x 1 x 1 x max_seq_len | ND |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | out_put  | float32  | batchsize x max_seq_len x 18000 | ND |





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
   # 默认当前目录为ACL_PyTorch/built-in/nlp/Pet_for_Pytorch
   cp *.py ${PaddleNLP}/example/few_shot/pet
   cp change.patch ${PaddleNLP}/example/few_shot/pet
   # 切换到paddleNLP Pet目录
   cd ${PaddleNLP}/example/few_shot/pet
   # 打patch
   git apply change.patch
   ```



2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   根据源码训练脚本训练模型后导出onnx模型，再使用atc工具导出om离线模型。

   1. 训练权重后导出onnx模型。
      推荐使用gpu环境训练
      
      通过如下命令，指定 GPU 0 卡,  在 FewCLUE 的 `eprstmt` 数据集上进行训练&评估
      ```
      python -u -m paddle.distributed.launch --gpus "0" run_train.py \
         --output_dir checkpoint_eprstmt \
         --task_name eprstmt \
         --split_id few_all \
         --prompt_path prompt/eprstmt.json \
         --prompt_index 0 \
         --do_train \
         --do_eval \
         --do_test \
         --do_predict \
         --do_label \
         --max_steps 1000 \
         --learning_rate 3e-5 \
         --eval_steps 100 \
         --save_steps 100 \
         --logging_steps 5  \
         --per_device_train_batch_size 16 \
         --max_seq_length 128 \
         --load_best_model_at_end \
         --metric_for_best_model accuracy \
         --save_total_limit 1 \
         --do_export \
         --export_type onnx
      ```
      参数含义说明
      - `task_name`: FewCLUE 中的数据集名字
      - `split_id`: 数据集编号，包括0, 1, 2, 3, 4 和 few_all
      - `prompt_path`: prompt 定义文件名
      - `prompt_index`: 使用定义文件中第 `prompt_index` 个 prompt
      - `augment_type`: 数据增强策略，可选 swap, delete, insert, substitute
      - `num_augment`: 数据增强策略为每个样本生成的样本数量
      - `word_augment_percent`: 每个序列中数据增强词所占的比例
      - `pseudo_data_path`: 使用模型标注的伪标签数据文件路径
      - `do_label`: 是否使用训练后的模型给无标签数据标注伪标签
      - `do_test`: 是否在公开测试集上评估模型效果
      - `model_name_or_path`: 预训练模型名，默认为 `ernie-1.0-large-zh-cw`
      - `use_rdrop`: 是否使用对比学习策略 R-Drop
      - `alpha_rdrop`: R-Drop 损失值权重，默认为 0.5
      - `dropout`: 预训练模型的 dropout 参数值，用于 R-Drop 策略中参数配置
      - `export_type`: 模型导出格式，默认为 `paddle`，动态图转静态图
       

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
               --output=Pet_bs${batch_size} \
               --input_format=ND \
               --input_shape="token_type_ids:${batch_size},128; \
                              input_ids:${batch_size},128; \
                              position_ids:${batch_size},128; \
                              attention_mask:${batch_size},1,1,128"\

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

           运行成功后生成`Pet_bs${batch_size}.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      数据集评估
      
      执行以下脚本测试数据集
      ```
      python evaluate_om.py \
      --task_name eprstmt \
      --split_id few_all \
      --prompt_path prompt/eprstmt.json \
      --prompt_index 0 \
      --max_seq_length 128 \
      --metric_for_best_model accuracy \
      --per_device_eval_batch_size ${batch_size} \
      --output_dir .
      --om_path Pet_bs${batch_size}.om \
      --device_id 0 \
      ```
      -   参数说明：
            - `task_name`: FewCLUE 中的数据集名字
            - `split_id`: 数据集编号，包括0, 1, 2, 3, 4 和 few_all
            - `prompt_path`: prompt 定义文件名
            - `prompt_index`: 使用定义文件中第 `prompt_index` 个 prompt
             -   `om_path`：om模型路径
             -   `device_id`：芯片ID
             -   `per_device_eval_batch_size`：模型bs
             - `max_seq_length`：最大输入长度

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

| 芯片型号 | Batch Size |数据集| 精度（Acc） | 性能（fps） |
| :------: | :--------:  | :--: | :--: | :--: |
|    310P3      |     1       |eprstmt|      88.5%    |  164.56    |
|    310P3      |     4       |      eprstmt|88.5%    |  305.86    |
|    310P3      |     8       |      eprstmt|88.5%    |    342.79  |
|    310P3      |     16       |    eprstmt|88.5%    |  320.80   |
|    310P3      |     32       |   eprstmt|88.5%    |  327.96    |
|    310P3      |     64       |      eprstmt|88.5%    |  329.28    |

测试tensorrt推理引擎，竞品性能参考下列数据。
| 芯片型号 | Batch Size  | 性能（fps） |
| :------: | :--------:  |  :--: |
|    T4      |     1           |  192.42    |
|    T4      |     4          |  272.46    |
|    T4      |     8           |    314.96  | 
|    T4      |     16           |  301.88    |
|    T4      |     32          |  294.39    |
|    T4      |     64           |  305.92    |
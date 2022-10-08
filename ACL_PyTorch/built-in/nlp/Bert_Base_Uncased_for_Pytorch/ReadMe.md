# 参考库文

-  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

# 参考实现

- ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
  branch=master
  commit_id=dd6b8ca2bb80e17b015c0f61e71c2a84733a5b32
  ```

# 依赖

| 依赖名称 | 版本   |
| -------- | :----- |
| ONNX     | 1.7.0  |
| Pytorch  | 1.6.0  |
| onnxsim  | 0.3.3  |
| boto3    | 1.21.1 |
| numpy    | 1.19.5 |

# 快速上手

#### 获取源码：

1. 获取github BERT源码

   

   下载代码仓。后续基本操作都是在DeepLearningExamples/PyTorch/LanguageModeling/BERT/目录下进行。

   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples
   git reset --hard dd6b8ca2bb80e17b015c0f61e71c2a84733a5b32
   ```

   

2. 下载modelzoo上源码包。

3. 上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

   

   ```
   ├── bert_config.json                 //bert_base模型网络配置参数
   ├── bert_base_get_info.py            //生成推理输入的数据集二进制info文件
   ├── bert_preprocess_data.py          //数据集预处理脚本，生成二进制文件
   ├── ReadMe.md
   ├── bert_base_uncased_atc.sh         //onnx模型转换om模型脚本
   ├── bert_base_pth2onnx.py            //用于转换pth模型文件到onnx模型文件
   ├── bert_postprocess_data.py         //bert_base数据后处理脚本，用于将推理结果处理映射成文本
   ├── add_attr_trans_b.py               //对可能存在的transpose进行优化
   └── evaluate_data.py                 //验证推理结果脚本，比对benchmark输出的分类结果，给出accuracy
   ```

   

   ![img](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/img/public_sys-resources/note_3.0-zh-cn.png)

   benchmark离线推理工具使用请参见《[CANN V100R020C20 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100180792)》

   

4. 将ModelZoo源码包中的文件移动并替换到DeepLearningExamples/PyTorch/LanguageModeling/BERT目录中。

#### 准备数据集

1. 获取原始数据集。

   

   本模型支持使用squad QA的验证集。以squad v1.1为例，请用户自行获取squad v1.1数据集，上传数据集到服务器目录DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/squad/v1.1/。

   

   

2. 数据预处理。

   

   

   数据预处理将原始数据集转换为模型输入的数据。模型输入数据为二进制输入。

   将原始数据（dev-v1.1.json文本）转化为二进制文件（.bin）。转化方法参考bert_preprocess_data.py训练预处理方法处理数据，以获得最佳精度，输出为二进制文件。

   执行bert_preprocess_data.py脚本。

   ```
   python3 bert_preprocess_data.py --max_seq_length=512 --do_lower_case --vocab_file=./vocab/vocab --predict_file=./data/squad/v1.1/dev-v1.1.json
   ```

   参数说明：

   - --max_seq_length：句子最大长度。
   - --vocab_file：数据字典映射表文件。
   - --do_lower_case：是否进行大小写转化。
   - --predict_file：原始验证数据文本，将后处理数据位置映射到原始文件。

   

3. 生成数据集info文件

   

   使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用bert_base_get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行bert_base_get_info.py脚本。

   ```
   python3 bert_base_get_info.py --batchsize=8
   ```

   参数为batchsize的大小，默认为8，运行成功后，在当前目录中生成bert_base_uncased.info。

   

#### 模型推理

1. ##### 模型转换。

   

   本模型基于开源框架PyTorch训练的bert_base_uncased进行模型转换。

   使用PyTorch将模型权重文件.pt转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. ###### 获取权重文件。

      - 在PyTorch开源框架中获取bert_base_qa.pt文件。

   2. ###### 导出onnx文件。

      将模型权重文件.pt转换为.onnx文件。

      1. 修改bert_base_qa.pt名称为pytorch_model.bin。

         ```
         mv bert_base_qa.pt pytorch_model.bin
         ```

         因为脚本读取已训练好的权重文件名为pytorch_model.bin。

      2. 进入BERT目录下，执行bert_base_pth2onnx.py脚本将.pt文件转换为.onnx文件，执行如下命令。

         ```
         python3.7 bert_base_pth2onnx.py --init_checkpoint=pytorch_model.bin --config_file=bert_config.json
         ```

         参数说明：

         - --init_checkpoint：输入权重文件。
         - --config_file：网络参数配置文件。

         运行成功后，在当前目录生成bert_base_batch_8.onnx模型文件。


         ![img](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/img/public_sys-resources/notice_3.0-zh-cn.png)
    
         使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为11。在bert_base_pth2onnx.py脚本中torch.onnx.export方法中的输入参数opset_version的值需为11，请勿修改。
    
      3. 此步可选，根据onnx图里是否存在（0，2，3，1）的transpose进行优化，若存在，运行下面命令。
    
         ```
         python3 add_attr_trans_b.py bert_base_batch_8.onnx bert_base_batch_8.onnx
         ```

   3. ###### 使用ATC工具将ONNX模型转OM模型。

      1. 修改bert_base_uncased_atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：

         ```
         # 配置环境变量
         source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
         
         # 使用二进制输入时，执行如下命令
         atc --input_format=ND --framework=5 --model=bert_base_batch_8.onnx --input_shape="input_ids:8,512;token_type_ids:8,512;attention_mask:8,512" --output=bert_base_batch_8_auto --log=info --soc_version=$1 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance --input_fp16_nodes="attention_mask"
         ```

         ![img](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/img/public_sys-resources/note_3.0-zh-cn.png)

         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。

          

         - 参数说明：
           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input_format：输入数据的格式。
           - --input_shape：输入数据的shape。
           - --log：日志等级。
           - --soc_version：部署芯片类型。

      2. 执行atc转换脚本，将.onnx文件转为离线推理模型文件.om文件。

         ${chip_name}可通过`npu-smi info`指令查看
         
         ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

         ```
         bash atc_bert_base_uncased.sh Ascend${chip_name} # Ascend310P3
         ```

         运行成功后生成bert_base_batch_8_auto.om用于二进制输入推理的模型文件。

   

2. ##### 开始推理验证。

   

   1. 使用Benchmark工具进行推理。

      增加benchmark.*{arch}可执行权限*。

      ```
      chmod u+x benchmark.x86_64
      ```

      验证模型性能：

      ```
      bash test_perf.sh
      ```

      获得**模型性能ave_throughputRate: 198.61samples/s**, 略微有点波动。

      执行命令，进行完整推理。

      ```
      bash infer_all.sh
      ```

      bert_base_uncased.info为处理后的数据集信息。

      benchmark.*{arch}*请根据运行环境架构选择，如运行环境为x86_64，需执行./benchmark.x86_64。推理后的输出默认在当前目录result下。

   2. 推理结果后处理

      将结果转化为json文本数据，执行命令如下。

      ```
      python3.7 bert_postprocess_data.py --max_seq_length=512 --vocab_file=./vocab/vocab --do_lower_case --predict_file=./data/squad/v1.1/dev-v1.1.json --npu_result=./result/dumpOutput/
      ```

      参数说明：

      - --max_seq_length：句子最大长度。
      - --vocab_file：数据字典映射表文件。
      - --do_lower_case：是否进行大小写转化。
      - --predict_file：原始验证数据文本，将后处理数据位置映射到原始文件。
      - --npu_result：benchmark推理结果目录。

   3. 精度验证

      调用evaluate_data.py脚本将原始数据dev-v1.1.json与推理结果数据文本predictions.json比对，可以获得Accuracy数据，结果保存在中，执行命令如下。

      ```
      python3.7 evaluate_data.py ./data/squad/v1.1/dev-v1.1.json predictions.json
      ```

      第一个参数为原始推理数据文本，第二个参数为生成推理结果文本。

      获得**模型精度：“f1”: 88.7402**
# Transformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

	- [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

	- [源码及环境](#section4622531142816)
	- [准备数据集](#section183221994411)
	- [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

传统占主导地位的序列转换模型是基于编码器-解码器的复杂递归网络或卷积神经网络实现的。表现最好的模型还通过注意机制连接编码器和解码器。论文作者提出了一种新的简单网络结构，transformer，完全基于注意机制，避免了重复和卷积。在两个机器翻译任务上的实验表明，这些模型具有更高的质量，同时具有更强的并行性和更少的训练时间。


- 参考实现：

  ```
  url=https://github.com/jadore801120/attention-is-all-you-need-pytorch
  branch=master
  commit_id=132907dd272e2cc92e3c10e6c4e783a87ff8893d
  ```
  说明：此模型未提供多个子模型，因此未填写model_name


  通过Git获取对应commit\_id的代码方法如下：

  ```bash
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小   | 数据类型 | 数据排布格式  |
  | -------- | ------ | -------- | ------------ |
  | input    | 1 x 15 | INT64    | ND           |


- 输出数据

  | 输出数据  | 大小   | 数据类型 | 数据排布格式  |
  | -------- | ------ | -------- | ------------ |
  | output   | 1 x 15 | INT64    | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 源码及环境<a name="section4622531142816"></a>

1. 下载开源项目attention-is-all-you-need-pytorch到服务器任意路径下。

   ```bash
   git clone https://github.com/jadore801120/attention-is-all-you-need-pytorch.git    # 克隆仓库的代码
   cd attention-is-all-you-need-pytorch    # 切换到模型的代码仓目录
   git reset --hard 132907dd272e2cc92e3c10e6c4e783a87ff8893d
   ```

2. 将ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/Transformer目录下的文件上传到服务器，并放到attention-is-all-you-need-pytorch源码目录下，后续操作均在开源项目attention-is-all-you-need-pytorch目录下进行。源码包中的文件及作用如下:

   **注意：开源项目attention-is-all-you-need-pytorch目录下也有`requirements.txt `，以ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/Transformerr目录下的`requirements.txt `为准。**

   ```
   ├── LICENSE                        // Apache LICENCE
   ├── modelzoo_level.txt             // 模型精度性能结果
   ├── README.md                      // 模型离线推理说明README
   ├── requirements.txt               // 环境依赖
   ├── Transformer_bleu_score.py      // 精度计算脚本
   ├── Transformer_ckpt2onnx.py       // 模型转换脚本
   ├── Transformer_modify_onnx.py     // 修改onnx脚本
   ├── Transformer_postprocess.py     // 模型后处理脚本
   ├── Transformer_preprocess.py      // 模型前处理脚本
   ```

3. 安装依赖。

   ```bash
   pip3 install -r requirements.txt   # 注意是ModelZoo-PyTorch仓中Transformer目录下的requirements.txt。约30分钟。
   python3 -m spacy download en      # 下载spacy语言模型
   python3 -m spacy download de
   ```
   说明：python3 -m spacy download en下载过慢或代理错误可以直接使用昇腾社区上已经下载好的模型。点击[链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Transformer/PTH/de-en.zip)，压缩包中de和en文件夹即为下载好的语言模型，将其拷贝到当前工作目录即可。

4. 安装改图依赖[auto-optimizer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)。

   ```
   # 安装改图工具auto-optimizer
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   python3 -m pip install .
   
   # 解决auto-optimizer依赖与requirements.txt中的冲突
   pip3 install click==7.1.2
   
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型采用torchtext中的Multi30k数据集，它是WMT 2016多模态任务小数据集，也称为Flickr30k。在文件attention-is-all-you-need-pytorch/preprocess.py中已使用torchtext.datasets.Multi30k这一api下载数据集，无需另行下载。

   torchtext.datasets.Multi30k会在preprocess.py文件同一目录下创建.data文件夹，下载并解压数据集压缩包。.data文件夹的目录结构如下：

   ```
   ├── .data  
         ├──mmt_task1_test2016.tar.gz // 3个压缩包都是torchtext.datasets.Multi30k下载的
         ├──test2016.de              // 其余文件都是解压缩之后的文件
         ├──test2016.en
         ├──test2016.fr
         ├──train.de
         ├──train.en
         ├──training.tar.gz
         ├──val.de
         ├──val.en
         ├──validation.tar.gz
   ```
   
2. 数据预处理。

   数据预处理将原始数据转化为二进制文件（.bin）输入。

   数据预处理共3步。
   
   1. 进入attention-is-all-you-need-pytorch源码目录。
   
   2. 对Multi30k数据集进行处理。其中preprocess.py是开源项目attention-is-all-you-need-pytorch下的文件，在“源码及环境”步骤已下载。
   
      ```bash
      mkdir ./pkl_file
      python3 preprocess.py -lang_src de -lang_trg en -share_vocab -save_data ./pkl_file/m30k_deen_shr.pkl
      ```
      - 参数说明：
         - -lang_src：源语言模型文件。
         - -lang_trg：目标语言模型文件。
         - -share_vocab：允许共享词典。
         - -save_data：pkl文件保存路径。

      运行成功后，生成/pkl_file/m30k_deen_shr.pkl文件（该命令若运行时被中断在重新运行时需要先删除.data文件夹）。

   3. 执行Transformer_preprocess.py脚本，把测试集数据转成bin文件（忽略测试集中长度大于15的句子）。

      ```bash
      mkdir -p ./pre_data/len15
      python3 Transformer_preprocess.py --src_lang=de --trg_lang=en --src_lang_mode_path=de --trg_lang_mode_path=en --dataset_parent_path=.data --pre_data_save_path=./pre_data/len15 --align_length 15
      ```

      - 参数说明：
        
         - --src_lang：源语言。
         - --trg_lang：目标语言。
         - --src_lang_mode_path：源语言模型文件路径。
         - --trg_lang_mode_path：目标语言模型文件路径。
         - --dataset_parent_path：Multi30k数据集路径。
         - --pre_data_save_path：预处理数据存放路径。
         - --align_length：数据对齐长度。
	
	   运行成功后，在/pre_data/len15目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型[权重文件.chkpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Transformer/PTH/transformer_trained_0.chkpt)转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 安装ais_bench推理工具

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 获取权重文件。

       本文以开源项目中的“WMT'16 Multimodal Translation: de-en”任务为例，训练好的权重文件transformer_trained_0.chkpt可从[链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Transformer/PTH/transformer_trained_0.chkpt)中获取。以model/transformer_trained_0.chkpt形式的目录结构放到当前工作目录。

   3. 导出onnx文件。

      在导出onnx文件之前，首先需要对开源仓中的代码进行2处更改，即下文中的1和2。

      1. 将源码中transformer/Translator.py文件中的Translator类换成以下代码。这是因为当前ATC不支持波束搜索算法，以下代码将其改为贪心搜索算法。
   
         ```python
         class Translator(nn.Module):
            ''' Load a trained model and translate in greedy search fashion. '''
         
            def __init__(
                     self, model, max_seq_len,
                     src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
               super(Translator, self).__init__()
         
               self.max_seq_len = max_seq_len
               self.trg_bos_idx = trg_bos_idx
               self.src_pad_idx = src_pad_idx
         
               self.model = model
               self.model.eval()
         
               self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
               self.register_buffer(
                     'blank_seqs',
                     torch.full((1, max_seq_len), trg_pad_idx, dtype=torch.long))
               self.blank_seqs[:, 0] = self.trg_bos_idx
         
            def forward(self, src_seq):
               src_mask = get_pad_mask(src_seq, self.src_pad_idx)
               enc_output, *_ = self.model.encoder(src_seq, src_mask)
               gen_seq = self.blank_seqs.clone().detach()
         
               trg_seq = self.init_seq
               for step in range(1, self.max_seq_len):
                     trg_mask = get_subsequent_mask(trg_seq)
                     dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
                     dec_output = F.softmax(self.model.trg_word_prj(dec_output), dim=-1)
         
                     _, best_k_idx = dec_output[0, -1, :].topk(1)
                     gen_seq[0, step] = best_k_idx
                     trg_seq = gen_seq[:, :step + 1]
         
               return gen_seq[0]
         ```
   
      2. 由于onnx不支持torch.triu()算子，因此需要修改源码中transformer/Models.py文件的get_subsequent_mask()函数。
   
         ```python
         def get_subsequent_mask(seq):
            ''' For masking out the subsequent info. '''
            sz_b, len_s = seq.size()
            # subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
            temp = np.triu(torch.ones((1, len_s, len_s)), 1)
            subsequent_mask = (1 - torch.from_numpy(temp)).bool()
            return subsequent_mask
         ```
   
      3. 运行Transformer_ckpt2onnx.py导出onnx模型。（用时约4分钟）
   
         ```bash
         python3 Transformer_ckpt2onnx.py -data_pkl ./pkl_file/m30k_deen_shr.pkl -model ./model/transformer_trained_0.chkpt -no_cuda -max_seq_len 15
         ```

         导出后的onnx模型为transformer_greedySearch_input15_maxSeqLen15.onnx。导出onnx名称可以在Transformer_ckpt2onnx.py文件中更改。
   
      4. 简化onnx模型（用时约10分钟）。
   
         ```bash
         python3 -m onnxsim ./model/transformer_greedySearch_input15_maxSeqLen15.onnx ./model/transformer_greedySearch_input15_maxSeqLen15_sim.onnx
         ```
         该命令中的两个路径，第一个路径为初始onnx路径，第二个路径为简化后的onnx的存储路径。

      5. 由于ATC的ScatterND和Slice算子不支持int64类型，GatherV2D算子的indices不支持“-1”输入，因此需要修改简化后的onnx模型。
         运行Transformer_modify_onnx.py脚本：
   
         ```bash
         python3 Transformer_modify_onnx.py --input_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim.onnx --output_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim_mod.onnx
         ```
         该命令所得到的结果./model/transformer_greedySearch_input15_maxSeqLen15_sim_mod.onnx，即为最终onnx，共经历导出、简化、修改三步。
   
      **注意：**
   
      - 使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为11。
      - 由于onnx模型文件大小限制，当前支持的最大翻译句子长度为15。
      - 此模型当前仅支持batch_size=1。
   
      
   
   4. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
      2. 执行命令查看芯片名称。
   
         ```
         npu-smi info
         # 该设备芯片名为Ascend310P3 
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.5         56                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            929  / 21534                            |
         +===================+=================+======================================================+
         
         ```
   
      3. 执行ATC命令。（用时约4分钟）
   
         ```bash
         # ATC转换onnx到om
         atc --framework=5 --model=./model/transformer_greedySearch_input15_maxSeqLen15_sim_mod.onnx --output=./model/transformer_greedySearch_input15_maxSeqLen15_finalom --input_format=ND --input_shape="input:1,15" --log=error --soc_version=Ascend310P3
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
         
         运行成功后生成 transformer_greedySearch_input15_maxSeqLen15_finalom.om 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  
      使用ais-bench之前需要配置环境变量：

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

      配置之后执行以下命令：

      ```bash
      python3 -m ais_bench --model ./model/transformer_greedySearch_input15_maxSeqLen15_finalom.om  --input ./pre_data/len15 --output ./result --output_dirname result_bs1
      ```
      - 参数说明：
         - --model：om模型路径
         - --input：数据预处理步骤最终生成二进制的文件夹
         - --output：推理结果输出目录。
         - --output_dirname：推理结果输出子文件夹

   2. 推理数据后处理。

      ```bash
      python3 Transformer_postprocess.py --bin_file_path ./result/result_bs1 --data_pkl ./pkl_file/m30k_deen_shr.pkl --result_path len15_ais_bench_result
      ```
      - 参数说明：
         - --bin_file_path：使用ais_bench推理工具进行推理时的output path，请注意修改。
         - --data_pkl：预处理之后的数据。
         - --result_path：存储后处理的结果的路径。

      后处理生成两个文件，其中pred_sentence.txt是翻译的句子，pred_sentence_array.txt是翻译句子对应的tensor值。

   3. 精度验证。

      ```bash
      python3 Transformer_bleu_score.py --ground_truth_file_path=./pre_data/len15/test_en_len15.txt --pred_file_path=./len15_ais_bench_result/pred_sentence.txt
      ```
      - 参数说明：
         - --ground_truth_file_path：标杆数据路径（在数据预处理的第3小步已根据测试数据集生成）。
         - --pred_file_path：ais_bench推理结果路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

| 芯片型号 | Batch Size | 数据集   | 精度（BLEU score） | 性能（FPS） |
| -------- | ---------- | -------- | ------------------ | ----------- |
| 310P     | 1          | Multi30k | 0.4098             | 65.4429     |

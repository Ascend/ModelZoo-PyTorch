# 欢迎使用昇腾推理引擎 AscendIE
昇腾推理引擎旨在提升AI模型迁移和应用开发的效率，并提升AI推理应用的性能。本部分内容包括推理引擎和框架推理插件。

该目录提供了基于昇腾推理引擎开展应用开发的参考样例，介绍主流的网络模型的迁移、推理的端到端流程，更多模型持续更新中。如果您有任何问题和需求，请在modelzoo/issues提交issue，我们会及时处理。


# 推理引擎
提供C++/Python统一接口，实现onnx 模型解析、构图等能力，快速实现ONNX模型向OM模型的迁移。详情使用请见[**AscendIE readme文档**](./AscendIE/readme.md)。

参考使用样例。

# 推理引擎算子
在CANN原生算子的基础上，推理引擎还针对特定模型，使用引擎自定义算子进行推理加速。

当前算子已支持Transformer类模型加速。这类模型通过减少推理过程中冗余计算进行加速，适用于长序列，且序列长度分布广的场景。

由于引擎暂未提供改图功能，Transformer类模型加速通过onnx改图和atc直接编译自定义算子的方式实现，请参考[bert动态模型优化](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Uncased_Huggingface), [albert动态模型转化](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/albert), [roberta动态模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/roberta)社区样例。

请注意当前Transformer类模型加速存在以下约束：
1. 推理批处理数量（batch size）不超过64
2. 序列长度（sequence length）为64的整数倍，且不超过1536
3. 注意力机制单头特征长度（embed size）为64

# 框架推理插件
提供C++/Python接口，少量代码完成Pytorch训练模型的迁移，实现高性能推理。

支持模型列表如下。

>**说明：**   
>**因使用版本差异，模型性能可能存在波动，性能仅供参考**


##  规范模型
CV-classfication

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>Top1Acc</td>
	<td>Top5Acc</td>
        <td>mAP</td>
    </tr>
    <tr>
        <td>
        <a href="https://github.com/pytorch/examples/tree/main/imagenet">  ResNet50 </a>
        </td>
        <td>ImageNet</td>
	<td>76.16%</td>
        <td>92.89%</td>
	<td></td>
	<td>2580(bs64)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
</table>



CV-detection

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>AP</td>
        <td>mAP</td>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://github.com/open-mmlab/mmocr">  DBNet </a>
        </td>
        <td>ICDAR2015</td>
	<td></td>
        <td></td>
        <td> </td>
	<td> </td>
	<td></td>
    </tr>
</table>


CV-recognition

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>AP</td>
        <td>mAP</td>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://github.com/open-mmlab/mmocr">  CRNN </a>
        </td>
        <td>svt</td>
	<td></td>
        <td></td>
        <td> 79.91% </td>
	<td> 1951(bs8) </td>
	<td>1 x 1 x 32 x wSize</td>
    </tr>
</table>



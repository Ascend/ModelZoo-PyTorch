# Vaecf

## 概述
- 模型使用框架

   **Cornac** 是多模式推荐系统的比较框架。它侧重于使利用**辅助数据**（例如，项目描述性文本和图像、社交网络等）的模型更方便地使用。Cornac支持**快速**实验和**直接**实施新模型。它与现有的机器学习库（例如 TensorFlow、PyTorch）**高度兼容**。

- 参考论文

    [Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara. “Variational Autoencoders for Collaborative Filtering.” arXiv:1802.05814v1](https://arxiv.org/pdf/1608.06993.pdf) 

- **Cornac**安装

  目前，我们支持 Python 3。有几种安装 Cornac 的方法：


  **1. From PyPI (you may need a C++ compiler):**
  ```bash
  pip3 install cornac
  ```

  **2. From Anaconda:**
  ```bash
  conda install cornac -c conda-forge
  ```

  **3. From the GitHub source (for latest updates):**
  ```bash
  pip3 install Cython
  git clone https://github.com/PreferredAI/cornac.git
  cd cornac
  python3 setup.py install
  ```



## 快速上手

- 数据集准备
1. 模型训练使用ml-20M数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，cornac框架里提供预处理的方法。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。


## 模型训练

- 根据cornac安装方式下载好cornac框架，并将代码仓内源码克隆下来。

- 启动训练之前，首先要配置程序运行相关环境变量。

    环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 运行源码根目录下的run.py文件即可开始训练。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

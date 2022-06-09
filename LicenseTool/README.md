## License添加工具使用说明：

#### 工具功能说明：

##### LICENSE文件添加：

1、网络根目录下若不存在LICENSE文件，会自行添加LICENSE文件。

2、若子目录存在多余LICENSE文件，会自行删除。

##### 脚本添加license注释：

1、在.py文件和.cpp文件开头自动判断是否存在license注释；若不存在，则会添加对应框架的license注释。

##### 注：

该工具当前仅支持tensorflow和pytorch两个框架下的迁移网络，mindspore和acl正在开发中。



#### 使用方法：

```shell
python3 LicenseTool.py --input_path XXXX路径
```

使用示例：

```shell
python3 LicenseTool.py --input_path /home/Bert-qa_ID0369_for_TensorFlow
```



注意事项：

1、因仓库统一格式，路径必须以`_for_TensorFlow`或`_for_PyTorch`结尾。

否则会出现报错：

`The name is unstandard ! Please use name like **_for_TensorFlow  or  **_for_PyTorch !`

2、`LicenseTool/data`目录为程序运行必须目录，请勿更改。


-   [AcessScan门禁扫描工具使用说明](#AcessScan门禁扫描工具使用说明.md)

-   门禁扫描checklist:

        1、license扫描：.py/.cpp文件中需要添加license，若未添加，结果返回失败；

        2、垃圾文件扫描：若后缀为.so/.log/.h5/.event/.log/.pbtxt/.zip/.tar/.tar.gz/.swp/.ipynp/.pyc及文件名称中含有.ckpt，则视为垃圾文件，结果返回失败；

        3、首层目录必要文件扫描：首层目录下必须存在LINCENSE,README.md,modelzoo_level.txt 三个文件，否则，结果返回失败；不可以存在00-access目录，否则，结果返回失

            败；（注：README.md可以命名为其他名称，如README.*/Readme.*/ReadMe.*/readme.*,*代表任意字符）

        4、大文件扫描：若文件大小超过2M，则视为大文件，结果返回失败； 

        5、内部链接扫描：文件中不可存在蓝驱无法访问的地址，如http://3ms.huawei.com,否则结果返回失败；（README.md不在扫描范围内） 

        6、敏感信息扫描：如工号，文件中的目录内容包含：wx+6/7/8/9位数字或00+6/7/8位数字的组合（扫描范围：所有文件）； 如IP：文件中不可存在IP，如：

            http://10.137.54.150/data,否则，结果返回失败(README.md不在扫描范围内)；

        7、网络功能、性能、精度扫描：网络首层目录下必须配置modelzoo_level.txt文件，且文件内容包含三个关键字段：FuncStatus(功能是否OK，值可填写OK/NOK)；

           PerfStatus(性能是否OK或达到极致PERFECT，值可填OK/NOK/PERFECT)；PrecisionStatus(精度是否OK,值可填写OK/NOK)；若网络功能、性能、精度均通过，内容格式如

           下所示： 

            
```
            FuncStatus:OK
            PerfStatus:OK
            PrecisionStatus:OK
```
            注：“:”两侧无需空格，英文格式；

            校验规则：

            a、FuncStatus:OK，PerfStatus:OK，PrecisionStatus:OK，表示网络功能、性能、精度均通过，网络代码必须放置在Official领域目录下；
        
            b、FuncStatus:OK，PerfStatus:PERFENCT，PrecisionStatus:OK，表示网络功能、精度均通过、性能达到极致，网络代码必须放置在Benchmark领域目录下；

            c、FuncStatus:OK，PerfStatus:NOK，PrecisionStatus:OK，表示网络功能通过、性能不通过、精度通过，网络代码必须放置在Research领域目录下；

            d、FuncStatus:OK，PerfStatus:OK，PrecisionStatus:NOK，表示网络功能通过、性能通过、精度不通过，网络代码必须放置在Research领域目录下；

            e、FuncStatus:OK，PerfStatus:NOK，PrecisionStatus:NOK，表示网络功能通过、性能不通过、精度不通过，网络代码必须放置在Research领域目录下；
    
            f、FuncStatus:NOK，表示网络功能不通过，网络代码不允许放置主仓内；
            

-   代码结构：

    
```
    ├── run_upline.sh //开始扫描执行脚本 
    ├── access_upline.py //实现门禁扫描规则的代码脚本 
    ├── link_list.txt //进行内部链接扫描时，内部链接关键字的配置脚本  
```


-   重要参数：

    -    run_upline.sh重要参数如下： 

        --id_dir //pr_filelist.txt 及modelzoo代码存放文件存放的路径

    -   access_upline.py重要参数如下：
 
        --pr_filelist_dir //需要上传仓上的所有文件名称及其路径（build-in开始的路径，例如：built-in/MindSpore/Benchmark/cv/detection/Mask_R_CNN_for_MindSpore/eval.py） 
        
        --linklisttxt //配置文件link_list.txt所在路径 

        --FileSizeLimit //配置大文件的大小，默认为2

-   操作步骤：(以AlexNet_for_TensorFlow为例) 

    1、获取代码：run_upline.sh、access_upline.py、link_list.txt脚本，下载脚本放置同一目录下，例如：/home 目录； 

    
```
    ├── /home 
    ├──├──run_upline.sh 
    ├──├──access_upline.py 
    ├──├──link_list.txt 
```


    2、数据准备：

        a、将需提交pr的代码放置在id_dir(可自定义),例如：/home: 
    

        ├── /home 
        ├──├──run_upline.sh 
        ├──├──access_upline.py 
        ├──├──link_list.txt 
        ├──├──modelzoo/build-in/... .../AlexNet_for_TensorFlow 

    
    
        b、pr_filelist.txt配置文件准备，pr_filelist.txt内容应该为AlexNet_for_TensorFlow目录下所有文件的路径如下所示： 
    
        built-in/TensorFlow/Official/cv/image_classification/AlexNet_for_TensorFlow/train.py 
        built-in/TensorFlow/Official/cv/image_classification/AlexNet_for_TensorFlow/README.md 
        built-in/TensorFlow/Official/cv/image_classification/AlexNet_for_TensorFlow/scripts/train_alexnet_1p.sh 
        built-in/TensorFlow/Official/cv/image_classification/AlexNet_for_TensorFlow/alexnet/alexnet.py 
        ...依次同理

         注意：路径是从build-in开始，且只需配置需要上传仓的文件，确保文件真实存在对应的路径下； 
    
        c、将pr_filelist.txt文件放置在modelzoo/目录同级，如下： 
    

        ├── /home 
        ├──├──run_upline.sh 
        ├──├──access_upline.py 
        ├──├──link_list.txt 
        ├──├──modelzoo/build-in/... .../AlexNet_for_TensorFlow 
        ├──├──pr_filelist.txt


    3、开始扫描 

        a、配置run_upline.sh脚本，将id_dir配置为实际路径，即/home id_dir='/home' 

        b、执行如下指令： bash run_upline.sh 

    4、扫描结果分析 

        a、若check success，则表示扫描通过,如下所示：
 
            
```
            =================Start to Check License ================= 
            =================Start to Check Size of File ================= 
            =================Start to Check funk file ================= 
            =================Start to Check file of First Directory ================= 
            =================Start to Check Internal Link ================= 
            =================Start to Check Sensitive Information ================= 
            =================Start to Check modelzoo level ================= 
            check success
```


        b、若结果返回 check fail,则表示失败，如下所示，失败原因请查看wiki门禁校验规则。

            
```
            =================Start to Check License ================= 
            =================Start to Check Size of File ================= 
            =================Start to Check funk file ================= 
            =================Start to Check file of First Directory ================= 
            =================Start to Check Internal Link ================= 
            =================Start to Check Sensitive Information ================= 
            =================Start to Check modelzoo level ================= 
            PerfStatus is not OK or PERFECT or PrecisionStatus is not OK ,You should put the code under the Research directory! 
            check fail
```

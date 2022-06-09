import sys
import os
import os.path
import filecmp
import argparse
import shutil
import gzip
import math
from pathlib import Path
import chardet
import re
import time
import json


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./AlexNet_for_TensorFlow",
                        help='model dirrectory of the project')
    parser.add_argument('--pr_filelist_dir', type=str, default="./pr_filelist0.txt",
                        help='model dirrectory of the pr_filelist')
    parser.add_argument('--linklisttxt', type=str, default='./link_list.txt',
                        help='model dirrectory of the link_list')
    parser.add_argument('--FileSizeLimit', type=int, default=2,
                        help='model size of FileSizeLimit')
    parser.add_argument('--train_performance_keyword', type=str, default="./train_performance_keyword.txt",
                        help='keywords of the train_performance_keyword file')
    parser.add_argument('--train_full_keyword', type=str, default="./train_full_keyword.txt",
                        help='keywords of the train_full_keyword file')
    return parser.parse_args()


# TODO 过于简单函数整改
def model_str_name(model_dir):
    '''
    功能：将路径转换为字符串
    '''
    model_dir_str = str(model_dir)
    return model_dir_str


# gitee上已经设置限制到文件到1M，这边可以设置更小，可以针对文件类型设置下
def file_size_check(path_pr_list, FileSizeLimit, fram_str, modelzoo_dir, dot=2, ):
    '''
    :param model_dir : 网络目录
    :param FileSizeLimit : 文件限制大小
    实现功能：扫描文件大小，小于2MB
    '''
    filesize_check = 0
    dict1 = {}
    with open(path_pr_list, 'r') as fooc:
        # 读取pr_filelist.txt的内容
        for model_dir in fooc:
            '''
                判断并处理三种类型的文件: LICENSE,py文件，其他文件
            '''
            model_dir = model_dir.strip('\n')
            # 获取模型框架路径字符串
            model_dir_str = model_str_name(model_dir)
            # 拼装路径
            model_dir2 = fram_str + modelzoo_dir + '/' + model_dir_str
            model_dir1 = Path(model_dir2)
            if model_dir1.exists():
                # 获取model_dir目录下所有文件
                pathTmp = str(model_dir1)
                filesize = os.path.getsize(pathTmp)  # 如果是文件，获取文件大小
                # 转换单位为兆
                filesize1 = str(round(filesize / math.pow(1024, 2), dot))
                dict1[model_dir_str] = filesize1  # 将文件大小添加到字典
            else:
                pass
        for key, value in dict1.items():
            if float(value) >= FileSizeLimit:
                print('{},size of file is {}M and greater than {}M,please check and delete it！'.format(key, value,
                                                                                                       FileSizeLimit))
                filesize_check = 1
            else:
                continue
        fooc.close()
    print('filesize_check=%d' % filesize_check)


def file_scan(path_pr_list, fram_str, modelzoo_dir):
    '''
        功能：判断.py/.cpp文件中是否存在关键字LICENSE/license，若不存在，则返回license_check状态为1，即失败；
        path_pr_list：pr_filelist.txt文件完整路径，其内容包含需要扫描文件
        fram_str：pr_filelist.txt文件所在的当前目录
        modelzoo_dir：modelzoo，字符串，用于拼接网络代码所在的完整路径
    '''
    with open(path_pr_list, 'r') as fooa:
        # 读取pr_filelist.txt的内容
        license_check = 0
        for model_dir in fooa:
            '''
                判断并处理三种类型的文件: LICENSE,py文件，其他文件
            '''
            # 去除换行符
            model_dir = model_dir.strip('\n')
            # 获取模型框架路径字符串
            model_dir_str = model_str_name(model_dir)
            # # 判断.py/.cpp文件中是否存在关键字LICENSE/license
            py_file = model_dir_str[-3:]
            cpp_file = model_dir_str[-4:]
            init_file = model_dir_str[-11:]
            if (py_file == '.py') or (cpp_file == '.cpp'):
                # 排除init文件
                if init_file == '__init__.py':
                    continue
                # 判断文件中是否存在关键字LICENSE/license
                else:
                    LICENSE, license, License, Licence, licence, LICENCE = 'LICENSE', 'license', 'License', 'Licence', 'licence', 'LICENCE'
                    model_dir1 = fram_str + modelzoo_dir + '/' + model_dir_str
                    model_dir = Path(model_dir1)
                    if model_dir.exists():
                        with open(str(model_dir), 'r', encoding='gb18030', errors='ignore') as foob:
                            content = foob.read()
                            if (LICENSE in content) or (license in content) or (License in content) or (
                                    Licence in content) or (licence in content) or (LICENCE in content):
                                continue
                            else:
                                license_check = 1
                                print('{},The keyword license no exists in the file,please check and add it!'.format(
                                    model_dir_str))
                            foob.close()
                    else:
                        pass
    print('license_check=%d' % license_check)
    if license_check == 1:
        print('License check failed, Please follow the guide to add License:')
        print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')

# TODO 过于简单函数整改
def spt_path(pr_filelist0_str):
    model_name_list = pr_filelist0_str.split('/')
    return model_name_list

# TODO 简化解析函数
def get_model_fram(pr_filelist0_str):
    '''
    功能：获取网络框架名称路径
    :param pr_filelist0_dir:
    :return:
    '''
    if 'for_TensorFlow' in pr_filelist0_str:
        if 'for_TensorFlow2.X' in pr_filelist0_str:
            tf_str = 'for_TensorFlow2.X'
            a = pr_filelist0_str.index(tf_str)
            b = a + 17
            model_fram = pr_filelist0_str[:b]
        else:
            tf_str = 'for_TensorFlow'
            a = pr_filelist0_str.index(tf_str)
            b = a + 14
            model_fram = pr_filelist0_str[:b]
    elif 'for_PyTorch' in pr_filelist0_str:
        pt_str = 'for_PyTorch'
        a = pr_filelist0_str.index(pt_str)
        b = a + 11
        model_fram = pr_filelist0_str[:b]
    elif 'for_MindSpore' in pr_filelist0_str:
        ms_str = 'for_MindSpore'
        a = pr_filelist0_str.index(ms_str)
        b = a + 13
        model_fram = pr_filelist0_str[:b]
    elif 'for_Tensorflow' in pr_filelist0_str:
        if 'for_Tensorflow2.X' in pr_filelist0_str:
            tf_str = 'for_Tensorflow2.X'
            a = pr_filelist0_str.index(tf_str)
            b = a + 17
            model_fram = pr_filelist0_str[:b]
        else:
            tf_str = 'for_Tensorflow'
            a = pr_filelist0_str.index(tf_str)
            b = a + 14
            model_fram = pr_filelist0_str[:b]
    elif 'for_Pytorch' in pr_filelist0_str:
        pt_str = 'for_Pytorch'
        a = pr_filelist0_str.index(pt_str)
        b = a + 11
        model_fram = pr_filelist0_str[:b]
    elif 'for_Mindspore' in pr_filelist0_str:
        ms_str = 'for_Mindspore'
        a = pr_filelist0_str.index(ms_str)
        b = a + 13
        model_fram = pr_filelist0_str[:b]
    elif 'for_ACL' in pr_filelist0_str:
        ms_str = 'for_ACL'
        a = pr_filelist0_str.index(ms_str)
        b = a + 7
        model_fram = pr_filelist0_str[:b]
    else:
        model_fram = ''
    return model_fram

# TODO 拆分功能
def check_firstlevel_file(path_pr_list, fram_str, modelzoo_dir):
    '''
        功能1：检查首层目录是否存在必要LICENSE文件
        功能2：检查首层目录是否存在必要README.md文件
        功能3：检查首层目录是否存在必要requirements.txt文件
        功能4：检查首层目录是否存在必要modelzoo_level.txt文件
        功能5：垃圾目录00-access拒绝入仓
        功能6：kernel_meta目录视为垃圾目录，拒绝入仓
        path_pr_list：pr_filelist.txt文件完整路径，其内容包含需要扫描文件
        fram_str：pr_filelist.txt文件所在的当前目录
        modelzoo_dir：modelzoo，字符串，用于拼接网络代码所在的完整路径
    '''
    # 不规范标识字段，0：pass,1:fail
    firstlevel_check = 0
    # readme检查
    firstlevel_check1 = 0
    # 00-access垃圾目录检查
    firstlevel_check2 = 0
    firstlevel_check6 = 0
    # LICENSE文件检查
    firstlevel_check3 = 0
    # modelzoo_level.txt检查
    firstlevel_check4 = 0
    # requirements.txt检查
    firstlevel_check5 = 0
    firstlevel_check8 = 0
    # kernel_metala垃圾目录
    firstlevel_check7 = 0
    with open(path_pr_list, 'r') as fooa:
        for filepath_inprlist in fooa:
            filepath_inprlist = filepath_inprlist.strip('\n')
            pr_filelist0_str = filepath_inprlist
            model_fram = get_model_fram(pr_filelist0_str)
            fram_path1 = fram_str + modelzoo_dir + '/' + model_fram
            fram_path = model_str_name(fram_path1)
            dir_str = model_str_name(filepath_inprlist)
            with open('first_filename4.txt', 'w') as file4:
                file4.write(str(dir_str))
                file4.close()
            with open('first_filename4.txt', 'r') as file5:
                content4 = file5.read()
                file5.close()
            h = 'requirements.txt'
            g = 'modelzoo_level.txt'
            # 如果网络名称不规范，排除推理及高校
            # if (model_fram == '' and 'built-in/ACL_' not in content4  and 'contrib' not in content4 ) :  # 网络名称不规范处理
            if (model_fram == ''):  # 网络名称不规范处理
                # 获取文件名
                fram_path2 = fram_str + modelzoo_dir + '/' + pr_filelist0_str
                file_name2 = os.path.basename(fram_path2)
                # 截取不规范网络名称路径
                b = filepath_inprlist.index(file_name2)
                fram_unst_dname = filepath_inprlist[:b]
                if fram_unst_dname != '':
                    fram_unst_dname_true = fram_str + modelzoo_dir + '/' + fram_unst_dname
                    filepath_inprlist2 = Path(fram_unst_dname_true)
                    if filepath_inprlist2.exists():
                        # 获取首层目录下所有文件
                        file_name3 = os.listdir(fram_unst_dname_true)
                        with open('first_filename3.txt', 'w') as file:
                            file.write(str(file_name3))
                            file.close()
                        with open('first_filename3.txt', 'r') as file2:
                            content2 = file2.read()
                            file2.close()
                        a, b, c, d, e = 'README', 'readme', 'LICENSE', 'Readme', 'ReadMe'
                        if (a in content2) or (d in content2) or (b in content2) or (e in content2):
                            if c in content2:
                                if g not in content2:
                                    firstlevel_check4 = 4
                                if h not in content2:
                                    firstlevel_check5 = 1
                        if '00-access' in content2:
                            firstlevel_check6 = 1
            if model_fram != '':
                # 判断路径是否真实存在
                filepath_inprlist1 = Path(fram_path)
                if filepath_inprlist1.exists():
                    # 获取首层目录下所有文件名称
                    a, b, c, d, e = 'README', 'readme', 'LICENSE', 'Readme', 'ReadMe'
                    filelist = os.listdir(fram_path)
                    with open('first_filename.txt', 'w') as file:
                        file.write(str(filelist))
                        file.close()
                    with open('first_filename.txt', 'r') as file1:
                        content1 = file1.read()
                        file1.close()
                    if '00-access' in content1:
                        firstlevel_check2 = 3
                    if (a not in content1) and (d not in content1) and (b not in content1) and (e not in content1):
                        firstlevel_check1 = 2
                        break
                    if c not in content1:
                        firstlevel_check3 = 4
                    if h not in content1:
                        firstlevel_check8 = 8
                        break

                else:
                    pass
            # kernel_meta目录视为垃圾目录，拒绝入仓
            if '/kernel_meta/' in content4:
                firstlevel_check7 = 1
        if firstlevel_check1 == 2:
            print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), a))
            firstlevel_check = 1
        if firstlevel_check8 == 8:
            print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), h))
            firstlevel_check = 1
        if firstlevel_check3 == 4:
            print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), c))
            print('License check failed, Please follow the guide to add LICENSE:')
            print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')
            firstlevel_check = 1
        if firstlevel_check4 == 4:
            print(
                '{},The {} file is non-existent in the model code of the file,Please follow the guide to add {}:'.format(
                    fram_unst_dname, g, g))
            print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')
            firstlevel_check = 1
        if firstlevel_check5 == 1:
            print('{},The {} file is non-existent in the model code of the file,Please check and add {}:'.format(
                fram_unst_dname, h, h))
            firstlevel_check = 1
        if firstlevel_check6 == 1:
            firstlevel_check = 1
            print('{},00-access directory should not exist,please delete it!'.format(fram_unst_dname))
        if firstlevel_check7 == 1:
            firstlevel_check = 1
            print('{}, kernel_meta is junk directory,please check and delete it!'.format(pr_filelist0_str))
        if firstlevel_check2 == 3:
            firstlevel_check = 1
            print('{},00-access directory should not exist,please delete it!'.format(get_model_fram(pr_filelist0_str)))
        print('firstlevel_check=%d' % firstlevel_check)

# TODO 不需要存在，只需要维护.gitignore即可
def junk_file(path_pr_list, fram_str, modelzoo_dir):
    '''
    功能:检测当前路径下所有垃圾文件
    参数:pr_filelist0_str:字符串化后的路径
    '''
    with open(path_pr_list, 'r') as fooa:
        funkfile_check = 0
        for filepath_inprlist in fooa:
            filepath_inprlist = filepath_inprlist.strip('\n')
            pr_filelist0_str1 = fram_str + modelzoo_dir + '/' + filepath_inprlist
            pr_filelist0_str = Path(pr_filelist0_str1)
            if pr_filelist0_str.exists():
                funk_file_typr = ['.log', '.pbtxt', '.pb', '.h5', '.so', '.zip', '.tar', '.event', '.tar.gz', '.swp',
                                  '.ipynb', '.pyc', '.novalocal', '.bin', '.pth', '.onnx', '.npy', '.om', '.pkl', '.pt',
                                  '.mat', '.tfrecord']
                junl_file_typr1 = ['.jpg', '.png']
                # 获取文件名
                file_name = os.path.basename(str(pr_filelist0_str))
                # 获取文件后缀名
                file_suffix = os.path.splitext(file_name)[1]
                if file_suffix in funk_file_typr:
                    funkfile_check = 1
                    print('{}, The file is Junk file, please check and delete it !'.format(filepath_inprlist))
                if '.ckpt' in str(pr_filelist0_str):
                    funkfile_check = 1
                    print('{}, The file is Junk file, please check and delete it !'.format(filepath_inprlist))
                if 'events.out.' in str(pr_filelist0_str):
                    funkfile_check = 1
                    print('{}, The file is Junk file, please check and delete it !'.format(filepath_inprlist))
                if 'network_need_files.txt' in str(pr_filelist0_str):
                    funkfile_check = 1
                    print('{}, The file is Junk file,please check and delete it!'.format(filepath_inprlist))
                # loss*.txt 视为垃圾文件
                loss_file_list = re.findall(r"\w*loss\w*.txt", file_name)
                loss_png_list = re.findall(r"\w*loss\w*.png", file_name)
                if loss_file_list != [] or loss_png_list != []:
                    funkfile_check = 1
                    print('{}, The file is Junk file,please check and delete it!'.format(filepath_inprlist))
                datas_path = ['data', 'datas', 'datasets']
                for data in datas_path:
                    # 目录中存在data目录
                    if data in str(filepath_inprlist):
                        data_str = data
                        path_list = filepath_inprlist.split('/')
                        for datapath in path_list:
                            if data_str in datapath:
                                a = filepath_inprlist.index(datapath)
                                fram_data = filepath_inprlist[:a]
                                # data所在路径
                                data_path = fram_str + modelzoo_dir + '/' + fram_data + datapath
                                # 如果路径真是存在
                                fram_data_str = Path(data_path)
                                if fram_data_str.exists():
                                    # 获取文件名称的后缀
                                    file_suffix1 = os.path.splitext(pr_filelist0_str1)[-1]
                                    if file_suffix1 in junl_file_typr1:
                                        # .jpg/.png在modelzoo下文件路径
                                        funkfile_check = 1
                                        print('{}, The file is Junk file,please check it!'.format(filepath_inprlist))
                                        break
                if 'ge_proto_' in str(pr_filelist0_str) and file_suffix == '.txt':
                    funkfile_check = 1
                    print('{}, The file is Junk file,please check it!'.format(filepath_inprlist))
                if 'events.' in str(pr_filelist0_str) and file_suffix == '.novalocal':
                    funkfile_check = 1
                    print('{}, The file is Junk file,please check it!'.format(filepath_inprlist))
            else:
                pass
        fooa.close()
        print('funkfile_check=%d' % funkfile_check)

# TODO 不需要存在，只需要维护.gitignore即可
def check_link(path_pr_list, fram_str, modelzoo_dir, onelink):
    '''
    功能：检测文件内部是否包含内部链接
    fram_file:文件所在路径
    alink: 一条字符串化的链接
    '''
    with open(path_pr_list, 'r') as fooa:
        internal_link_check = 0
        for filepath_inprlist in fooa:
            filepath_inprlist = filepath_inprlist.strip('\n')
            pr_filelist0_str1 = fram_str + modelzoo_dir + '/' + filepath_inprlist
            # 将路径名称字符串化
            file_name = model_str_name(pr_filelist0_str1)
            if ('README' not in file_name) and ('readme' not in file_name) and ('Readme' not in file_name) and (
                    'ReadMe' not in file_name):
                pr_filelist0_str = Path(file_name)
                if pr_filelist0_str.exists():
                    with open(str(pr_filelist0_str), 'r', encoding='gb18030', errors='ignore') as foo:
                        for words in foo:
                            if onelink in words:
                                link = onelink[0:]
                                internal_link_check = 1
                                print(
                                    '{},This is an internal links that includes {},please check this line that: {}'.format(
                                        filepath_inprlist, link, words))
                            else:
                                continue
                        foo.close()
                else:
                    pass
        print('internal_link_check=%d' % internal_link_check)

# TODO 含义复制，需要重构
def check_Sensitive_content(path_pr_list, fram_str, modelzoo_dir):
    with open(os.getcwd() + "/upline_access_black_http.json", 'r') as load_f:
        load_dict = json.load(load_f)
    with open(path_pr_list, 'r') as fooa:
        sensitive_check = 0
        for fram_file_dir in fooa:
            # 去除换行操作
            fram_file_dir = fram_file_dir.strip('\n')
            # pr中文件绝对路径
            pr_filelist0_str1 = fram_str + modelzoo_dir + '/' + fram_file_dir
            # 判断文件是否存在
            file_name = model_str_name(pr_filelist0_str1)
            pr_filelist0_str = Path(file_name)
            if pr_filelist0_str.exists():
                # 如果文件存在，打开文件
                with open(str(pr_filelist0_str), 'r', encoding='gb18030', errors='ignore') as foo:
                    for words in foo:
                        words = words.strip('\n')
                        if ('0.00' not in words) and ('0.' not in words):
                            # 工号识别
                            if re.findall(
                                    r'([/][A-Za-z]00[\d]{5}[/]|[/][A-Za-z]00[\d]{6}[/]|[/][A-Za-z]00[\d]{7}[/]|[/][A-Za-z]00[\d]{8}[/]|[/][A-Za-z]00[\d]{9}[/]|[/][A-Za-z]00[\d]{10}[/])',
                                    words) or \
                                    re.findall(
                                        r'([/][A-Za-z]wx\d{6}[/]|[/][A-Za-z]wx\d{7}[/]|[/][A-Za-z]wx\d{8}[/]|[/][A-Za-z]wx\d{9}[/]|[/][A-Za-z]wx\d{10}[/]|[/][A-Za-z]wx\d{11}[/])',
                                        words) or \
                                    re.findall(
                                        r'([/]00[\d]{5}[/]|[/]00[\d]{6}[/]|[/]00[\d]{7}[/]|[/]00[\d]{8}[/]|[/]00[\d]{9}[/]|[/]00[\d]{10}[/])',
                                        words):
                                print(
                                    '{}, There may be a job number in the file, please check the line that is: {}'.format(
                                        fram_file_dir, words))
                                sensitive_check = 1
                    foo.close()
                # 获取文件名
                env_file_name = os.path.basename(str(pr_filelist0_str))
                if re.findall(r'train\w*.sh', env_file_name) or re.findall(r'infer\w*.sh', env_file_name):
                    with open(str(pr_filelist0_str), 'r', encoding='gb18030', errors='ignore') as fooc:
                        for words in fooc:
                            words = words.strip('\n')
                            # 不合规配置环境变量1、export install_path 2、export LD_LIBRARY_PATH 3、export PYTHONPATH 4、export PATH 5、export ASCEND_OPP_PATH
                            if re.findall(r'\w*export install_path\w*', words) or re.findall(
                                    r'\w*export LD_LIBRARY_PATH\w*', words) or re.findall(r'\w*export PATH\w*', words) \
                                    or re.findall(r'\w*export ASCEND_OPP_PATH\w*', words):
                                print(
                                    '{}, There are non compliant configuration environment variables, please check the line that is: {}'.format(
                                        fram_file_dir, words))
                                sensitive_check = 1
                            if re.findall(r'\w*export PYTHONPATH\w*', words) and re.findall(r'\w*install_path\w*',
                                                                                            words):
                                print(
                                    '{}, There are non compliant configuration environment variables, please check the line that is: {}'.format(
                                        fram_file_dir, words))
                                sensitive_check = 1
                        fooc.close()
                if ('README' not in file_name) and ('readme' not in file_name) and ('Readme' not in file_name) and (
                        'ReadMe' not in file_name):
                    with open(str(pr_filelist0_str), 'r', encoding='gb18030', errors='ignore') as fooc:
                        for words in fooc:
                            # ip 识别
                            if ('device_ip' in words) or ('server_id' in words):
                                continue
                            elif re.findall(
                                    r'http://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                                    words) or \
                                    re.findall(
                                        r'https://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                                        words):  # or \
                                # re.findall(
                                #     r'(?<![\.\d])(?:25[0-5]\.|2[0-4]\d\.|[01]?\d\d?\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\.\d])',
                                #     words) or \
                                # re.findall(
                                #     r'\b(?:25[0-5]\.|2[0-4]\d\.|[01]?\d\d?\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
                                #     words):
                                print(
                                    '{}, There may be an ip address in the file, please check the line that is:{}'.format(
                                        fram_file_dir, words))
                                sensitive_check = 1
                        fooc.close()
                # if 'contrib/' not in str(fram_file_dir):
                # 数据集合规性检查、扫描数据集版权合规
                if ('README' in file_name) or ('readme' in file_name) or ('Readme' in file_name) or (
                        'ReadMe' in file_name):
                    with open(str(pr_filelist0_str), 'r', encoding='gb18030', errors='ignore') as fooc:
                        for words in fooc:
                            blackhttp = load_dict["blackhttp"]
                            blackhttp_list = blackhttp.split(",")
                            for black_key in blackhttp_list:
                                if 'contrib/' in str(fram_file_dir) and 'pan.baidu.com' == black_key:
                                    pass
                                else:
                                    if black_key in words:
                                        print(
                                            '{}, The dataset used is not copyright compliant in the file, please check the line that is: {}'.format(
                                                fram_file_dir, words))
                                        sensitive_check = 1
        print('sensitive_check=%d' % sensitive_check)

# TODO 魔鬼变量过多，需要整改，需要更加表意
def modelzoo_level_check(path_pr_list, fram_str, modelzoo_dir):
    # 获取首层目录下所有文件
    with open(path_pr_list, 'r') as fooa:
        for filepath_inprlist in fooa:
            # 去除换行
            filepath_inprlist = filepath_inprlist.strip('\n')
            # 默认功能返回结果
            modelzoo_level_check = 0
            modelzoo_level_check1 = 0
            modelzoo_level_check2 = 0
            modelzoo_level_check3 = 0
            modelzoo_level_check4 = 0
            modelzoo_level_check5 = 0
            modelzoo_level_check6 = 0
            modelzoo_level_check7 = 0
            pr_filelist0_str = filepath_inprlist
            model_fram = get_model_fram(pr_filelist0_str)
            if model_fram == '' and 'built-in' in pr_filelist0_str and 'Common' not in pr_filelist0_str:  # 高校网络名称不规范处理
                modelzoo_level_check6 = 1
                print('{},Network name of the file naming is not standard, please check and modify it!'.format(
                    pr_filelist0_str))
                break
                # 配置需要读取的文件
            a, b, c, d = 'modelzoo_level.txt', 'FuncStatus', 'PerfStatus', 'PrecisionStatus'
            # else:#网络名称符合规范处理
            if model_fram != '':
                # 线上实际网络代码首层路径
                fram_path1 = fram_str + modelzoo_dir + '/' + model_fram
                fram_path = model_str_name(fram_path1)
                # 判断路径是否存在
                filepath_inprlist = Path(fram_path)
                if filepath_inprlist.exists():
                    # 获取首层目录下所有文件名称
                    filelist = os.listdir(fram_path)
                    # 所有文件名称写入level_filename.txt文件中
                    with open('level_filename_line.txt', 'w') as file:
                        file.write(str(filelist))
                        file.close()
                    with open('level_filename_line.txt', 'r') as file1:
                        content1 = file1.read()
                        file1.close()
                    # 判断modelzoo_level.txt是否存在
                    level_dict = {}
                    levelfile_path = fram_path1 + '/' + a
                    pr_levelfile_path = get_model_fram(pr_filelist0_str) + '/' + a
                    if a in content1 and Path(fram_path).exists():
                        # 打开modelzoo_level.txt文件，读取内容
                        with open(levelfile_path, 'r', encoding='gb18030', errors='ignore') as fooe:
                            content2 = fooe.read()
                            fooe.close()
                        # 判断#modelzoo_level.txt文件中是否在关键字段
                        if (b in content2) and (c in content2) and (d in content2):
                            # 读取modelzoo_level.txt内容，截取关键字段及其值
                            try:
                                fopen = open(levelfile_path)
                                for line in fopen.readlines():
                                    line = str(line).replace("\n", "")  # 注意，必须是双引号，找了大半个小时，发现是这个问题。。
                                    level_dict[line.split(':', 1)[0]] = line.split(':', 1)[1]
                                    # split（）函数用法，逗号前面是以什么来分割，后面是分割成n+1个部分，且以数组形式从0开始
                                fopen.close()
                            except:
                                print(
                                    'When Check Modelzoo Level,Return ERROR,Please check contents of modelzoo_level.txt ')
                                modelzoo_level_check = 1
                                break
                            # 如果#modelzoo_level.txt文件中存在关键字段，判断字段是否为空
                            if level_dict[str('FuncStatus')] != '' and level_dict[str('PerfStatus')] != '' and \
                                    level_dict[str('PrecisionStatus')] != '':
                                # 功能、性能、精度都OK，网路归属Official
                                if level_dict[str('FuncStatus')] != 'NOK' and level_dict[
                                    str('PrecisionStatus')] != 'NOK':
                                    # 高校不校验
                                    if level_dict[str('FuncStatus')] == 'OK' and level_dict[
                                        str('PerfStatus')] == 'OK' and level_dict[str('PrecisionStatus')] == 'OK':
                                        if 'Official' in fram_path:
                                            pass
                                        else:
                                            modelzoo_level_check4 = 1
                                    elif level_dict[str('FuncStatus')] == 'OK' and level_dict[
                                        str('PerfStatus')] == 'NOK' and level_dict[str('PrecisionStatus')] == 'OK':
                                        if 'Research' in fram_path:
                                            pass
                                        else:
                                            modelzoo_level_check2 = 1
                                    # 功能、精度都OK，性能为PERFECT,网路归属Benchmark
                                    elif level_dict[str('FuncStatus')] == 'OK' and level_dict[
                                        str('PerfStatus')] == 'PERFECT' and level_dict[str('PrecisionStatus')] == 'OK':
                                        if 'Benchmark' in fram_path:
                                            pass
                                        else:
                                            modelzoo_level_check1 = 1
                                    # 功能OK，性能not PERFECT OR OK ,精度 not OK ,网络归属Research目录下
                                    else:
                                        if 'Research' in fram_path:
                                            pass
                                        else:
                                            modelzoo_level_check2 = 1
                                # 功能not OK,网路归属非Modelzoo领域
                                else:
                                    if level_dict[str('FuncStatus')] == 'NOK' and 'built-in' in pr_filelist0_str:
                                        modelzoo_level_check3 = 1
                                        print(
                                            '{},FuncStatus is NOK ,You can not put the code under the directory of modelzoo!'.format(
                                                get_model_fram(pr_filelist0_str)))
                                    if level_dict[str('PrecisionStatus')] == 'NOK' and 'built-in' in pr_filelist0_str:
                                        modelzoo_level_check5 = 1
                                        print(
                                            '{},PrecisionStatus is NOK ,You can not put the code under the directory of modelzoo!'.format(
                                                get_model_fram(pr_filelist0_str)))
                            else:
                                if level_dict[str('FuncStatus')] == '':
                                    modelzoo_level_check5 = 1
                                    print(
                                        '{},The value of the FuncStatus is null in the file,please check and enter the correct value!'.format(
                                            pr_levelfile_path))
                                if level_dict[str('PerfStatus')] == '':
                                    modelzoo_level_check5 = 1
                                    print(
                                        '{},The value of the PerfStatus is null in the file,please check and enter the correct value!'.format(
                                            pr_levelfile_path))
                                if level_dict[str('PrecisionStatus')] == '':
                                    modelzoo_level_check5 = 1
                                    print(
                                        '{},The value of the PrecisionStatus is null in the file,please check and enter the correct value!'.format(
                                            pr_levelfile_path))
                        else:
                            if b not in content2:
                                modelzoo_level_check5 = 1
                                print(
                                    '{},The keyword of the FuncStatus is not exist in the file,please check and add it!'.format(
                                        pr_levelfile_path))
                            if c not in content2:
                                modelzoo_level_check5 = 1
                                print(
                                    '{},The keyword of the PerfStatus is not exist in the file,please check and  add it!'.format(
                                        pr_levelfile_path))
                            if d not in content2:
                                modelzoo_level_check5 = 1
                                print(
                                    '{},The keyword of the PrecisionStatus is not exist in the file,please check and add it!'.format(
                                        pr_levelfile_path))
                    else:
                        modelzoo_level_check7 = 1
            if modelzoo_level_check7 == 1:
                modelzoo_level_check = 1
                print('{},The {} file is not exist,Please follow the guide to add {}:'.format(
                    get_model_fram(pr_filelist0_str), a, a))
                print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')
                break
            if modelzoo_level_check4 == 1 and 'built-in' in pr_filelist0_str:
                modelzoo_level_check = 1
                print(
                    '{},All FuncStatus PerfStatus PrecisionStatus are OK ,You should put the code under the Official directory!'.format(
                        get_model_fram(pr_filelist0_str)))
                break
            if modelzoo_level_check3 == 1 and 'built-in' in pr_filelist0_str:
                modelzoo_level_check = 1
                break
            if modelzoo_level_check2 == 1 and 'built-in' in pr_filelist0_str:
                modelzoo_level_check = 1
                print(
                    '{},The optimal performance or accuracy are not completely achieved by studying the class model ,You should put the code under the Research directory!'.format(
                        get_model_fram(pr_filelist0_str)))
                break
            if modelzoo_level_check1 == 1 and 'built-in' in pr_filelist0_str:
                modelzoo_level_check = 1
                print(
                    '{},PerfStatus is PERFECT and FuncStatus PrecisionStatus are OK ,You should put the code under the Benchmark directory!'.format(
                        get_model_fram(pr_filelist0_str)))
                break
            if modelzoo_level_check5 == 1:
                modelzoo_level_check = 1
                print('Please follow the guide (NO.5 PR submission) to self-checking :')
                print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')
                break
            if modelzoo_level_check6 == 1:
                print('Please follow the guide to self-checking :')
                print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')
                break
    print('modelzoo_level_check=%d' % modelzoo_level_check)

# TODO 无效代码过多，需要整改
def file_word_check(fram_str, modelzoo_dir, path_pr_list, train_full_keyword, train_performance_keyword):
    a, b = 'train_performance_1p.sh', 'train_full'
    c, d = [], []
    with open(path_pr_list, 'r') as fooa:
        for filepath_inprlist in fooa:
            # 去除换行
            filepath_inprlist = filepath_inprlist.strip('\n')
            # 默认功能返回结果
            file_word_check = 0
            file_word_check7 = 0
            pr_filelist0_str = filepath_inprlist
            model_fram = get_model_fram(pr_filelist0_str)
            if model_fram == '':  # 高校网络名称不规范处理
                continue
            else:  # 网络名称符合规范处理
                # 线上实际网络代码首层路径
                fram_path1 = fram_str + modelzoo_dir + '/' + get_model_fram(pr_filelist0_str)
                fram_path = model_str_name(fram_path1)
                # 判断路径是否存在
                filepath_inprlist = Path(fram_path)
                if filepath_inprlist.exists():
                    # 获取网络首层目录下所有目录名称列表
                    fram_allfilename = os.listdir(fram_path)
                    for test_dir in fram_allfilename:
                        if 'test' == test_dir:
                            # 获取网络路径与其下的目录文件名的拼接路径
                            test_path = fram_path1 + '/' + 'test'
                            test_allfilename = os.listdir(test_path)
                            # 所有文件名称写入perfm_full.txt文件中
                            with open('perfm_full.txt', 'w') as file:
                                file.write(str(test_allfilename))
                                file.close()
                            with open('perfm_full.txt', 'r') as file1:
                                content1 = file1.read()
                            # 过滤推理
                            if 'built-in/ACL_' not in fram_path and 'contrib/ACL_' not in fram_path:
                                if b in content1:
                                    # if a in test_allfilename:
                                    #     performance_path = test_path + '/' + a
                                    #     with open(performance_path,'r',errors='ignore') as train_performance:
                                    #         performance_contents = train_performance.read()
                                    #         train_performance.close()
                                    #         file_word_check3 = 1
                                    # else:
                                    #     file_word_check6 = 1

                                    # if b in test_allfilename:
                                    #     full_path = test_path + '/' + b
                                    #     file_word_check5 = 1
                                    #     with open(full_path,'r',errors='ignore') as train_full:
                                    #         full_contents=train_full.read()
                                    #         train_full.close()
                                    continue
                                    # full_path = test_path + '/' + b
                                    # file_word_check5 = 1
                                    # with open(full_path,'r',errors='ignore') as train_full:
                                    #     full_contents=train_full.read()
                                    #     train_full.close()
                                else:
                                    file_word_check7 = 1
                                    break
            if file_word_check7 == 1:
                file_word_check = 1
                print('{},The train_full(1p or 8p) file is not exist,please check and add it!'.format(
                    get_model_fram(pr_filelist0_str) + '/test'))
                break
            # if file_word_check6 == 1:
            #     file_word_check = 1
            #     print('{},The file is not exist,please add it!'.format(get_model_fram(pr_filelist0_str) + '/' + a))
            #     break
            # train_performance_1p.sh脚本关键字校验
            # if file_word_check3 == 1:
            #     with open(train_performance_keyword,'r',errors='ignore') as foo_trainperformancekeyword: #打开关键字文件
            #         for performance_keyword in foo_trainperformancekeyword:
            #             performance_keyword = performance_keyword.strip('\n')
            #             if performance_keyword in performance_contents:
            #                 c.append(performance_keyword)
            #         foo_trainperformancekeyword.close()
            #     if len(c) < 8 :
            #         file_word_check1 = 1
            # ##train_full_1p.sh脚本关键字校验
            # if file_word_check5 == 1:
            #     with open(train_full_keyword, 'r', errors='ignore') as foo_trainfullword:  # 打开关键字文件
            #         for full_keyword in foo_trainfullword:
            #             full_keyword = full_keyword.strip('\n')
            #             if full_keyword in full_contents:
            #                 d.append(full_keyword)
            #         foo_trainfullword.close()
            #         if len(d) < 9:
            #             file_word_check2 = 1
            # if file_word_check4 == 1 :
            #     print('=================Start to Check File&Keywords of Test Directory  =================')
            # if file_word_check1 == 1:
            #     file_word_check = 1
            #     print('{},The file missing keywords,Please check it!'.format(get_model_fram(pr_filelist0_str) + '/' + a))
            #     break
            # if file_word_check2 == 1:
            #     file_word_check = 1
            #     print('{},The file missing keywords,Please check it!'.format(get_model_fram(pr_filelist0_str) + '/' + b))
            #     break
        print('file_word_check=%d' % file_word_check)

# TODO 绑核检查，未必合理，需要整改
def scan_core_binding(path_pr_list, fram_str, modelzoo_dir):
    json_file = open("core_binding_config.json")
    load_dict = json.load(json_file)
    judge = list(load_dict.keys())
    # 默认功能返回结果
    core_binding_check = 0
    with open(path_pr_list, 'r') as fooa:
        for filepath_inprlist in fooa:
            # 去除换行
            filepath_inprlist = filepath_inprlist.strip('\n')
            # 路径字符串化
            pr_filelist0_str = str(filepath_inprlist)
            filefull_path = fram_str + modelzoo_dir + '/' + pr_filelist0_str
            fullfile_path = model_str_name(filefull_path)
            i, y = 0, 0
            # 判断路径是否存在
            filepath_inprlist = Path(fullfile_path)
            if filepath_inprlist.exists():
                if '.sh' in fullfile_path:
                    for n in range(0, judge.__len__()):
                        need_param = load_dict[judge[n]]["need"].split(",")
                        notneed_param = load_dict[judge[n]]["not need"].split(",")
                        with open(filefull_path, 'r', encoding='gb18030', errors='ignore') as files:
                            file_contents = files.read()
                            files.close()
                        for line in open(fullfile_path, encoding='gb18030', errors='ignore'):
                            if '#' not in line and 'taskset -c' in line:
                                # need_param[0]=taskset -c $
                                if need_param[0] not in line:
                                    i = 1
                                    core_binding_check = 1
                                if notneed_param[0] in line:
                                    if ('a=RANK_ID*$' not in file_contents) and ('a=$' not in file_contents) and (
                                            'a=RANK_ID_n*$' not in file_contents):
                                        i = 1
                                        core_binding_check = 1
                            if "RANK_TABLE_FILE" in line and "/1p.json" in line and "#" not in line:
                                y = 1
                                core_binding_check = 1
                if "1p.json" in fullfile_path or "8p.json" in fullfile_path:
                    for line in open(fullfile_path, 'r', encoding='utf-8'):
                        # 判断board_id是否存在json文件中
                        if "board_id" in line:
                            y = 1
                            core_binding_check = 1
                if y == 1:
                    print('{},Device ID is written dead in the file, please check and modify it'.format(
                        pr_filelist0_str))
                if i == 1:
                    print('{},The file has binding cores, please check and modify it'.format(pr_filelist0_str))
    print('core_binding_check=%d' % core_binding_check)

# 无效代码删除，对应每个检查部分需要做好注释工作，另外硬编码较多，需要看看是否合理
def main():
    args = init_args()
    path_pr_list = args.pr_filelist_dir
    alink = args.linklisttxt
    train_full_keyword = args.train_full_keyword
    train_performance_keyword = args.train_performance_keyword
    tf_str = 'pr_filelist.txt'
    a = path_pr_list.index(tf_str)
    fram_str = path_pr_list[:a]
    modelzoo_dir = 'modelzoo'
    FileSizeLimit = args.FileSizeLimit
    print('=================Start to Check License =================')
    file_scan(path_pr_list, fram_str, modelzoo_dir)

    # file_scan_time_end = time.time()
    # file_scan_time = file_scan_time_end - file_scan_time_start
    # print('Check License time is {}'.format(file_scan_time))
    '''
    :param model_dir : 网络目录
    :param FileSizeLimit : 文件限制大小
    实现功能：扫描文件大小，小于2MB
    '''
    print('=================Start to Check Size of File =================')
    file_size_check(path_pr_list, FileSizeLimit, fram_str, modelzoo_dir, dot=2)
    print('=================Start to Check Junk file  =================')
    junk_file(path_pr_list, fram_str, modelzoo_dir)
    # 层级目录检查
    print('=================Start to Check file of First Directory =================')
    check_firstlevel_file(path_pr_list, fram_str, modelzoo_dir)
    print('=================Start to Check Internal Link =================')
    with open(alink, 'r') as food:
        for onelink in food:
            onelink = onelink.strip('\n')
            check_link(path_pr_list, fram_str, modelzoo_dir, onelink)
    print('=================Start to Check Sensitive Information =================')
    check_Sensitive_content(path_pr_list, fram_str, modelzoo_dir)
    print('=================Start to Check Modelzoo Level =================')
    # TODO official已删，不需要再检查放哪了，modelzoo_level检查可以继续保留
    # modelzoo_level_check(path_pr_list, fram_str, modelzoo_dir)
    print('=================Start to Check File&Keywords of Test Directory  =================')
    file_word_check(fram_str, modelzoo_dir, path_pr_list, train_full_keyword, train_performance_keyword)
    print('=================Start to Check core_binding&Device Id status =================')
    scan_core_binding(path_pr_list, fram_str, modelzoo_dir)


if __name__ == '__main__':
    main()

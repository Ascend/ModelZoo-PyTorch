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
    return parser.parse_args()


def model_str_name(model_dir):
    '''
    功能：将路径转换为字符串
    '''
    model_dir_str = str(model_dir)
    return model_dir_str


def file_size_check(path_pr_list, FileSizeLimit, fram_str, modelzoo_dir, dot=2, ):
    '''
    :param model_dir : 网络目录
    :param FileSizeLimit : 文件限制大小
    实现功能：扫描文件大小，小于2MB
    '''
    filesize_check = 0
    # path = fram_str
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
                # print(pathTmp)
                filesize = os.path.getsize(pathTmp)  # 如果是文件，获取文件大小
                # 转换单位为兆
                filesize1 = str(round(filesize / math.pow(1024, 2), dot))
                # print('{} 文件大小为:{}'.format(filename, filesize))
                dict1[model_dir_str] = filesize1  # 将文件大小添加到字典
            else:
                print('{},The file is not exist!'.format(model_dir))
                filesize_check = 0
        for key, value in dict1.items():
            if float(value) >= FileSizeLimit:
                print('{},size of file is {}M and  greater than {}M,please check it！'.format(key, value, FileSizeLimit))
                filesize_check = 1
            else:
                continue
    print('filesize_check=%d' % filesize_check)


def file_scan(path_pr_list, fram_str, modelzoo_dir):
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
            # # 处理LICENSE文件
            # # 判断LICENSE文件中是否存在关键字LICENSE/license
            # # 判断.py/.cpp文件中是否存在关键字LICENSE/license
            a = model_dir_str[-3:]
            b = model_dir_str[-7:]
            c = model_dir_str[-4:]
            z = model_dir_str[-11:]
            if (b == 'LICENSE') or (a == '.py') or (c == '.cpp'):
                # 判断LICENSE文件中是否存在关键字LICENSE/license
                if z == '__init__.py':
                    continue
                else:
                    a, b, c, d, e, f = 'LICENSE', 'license', 'License', 'Licence', 'licence', 'LICENCE'
                    model_dir1 = fram_str + modelzoo_dir + '/' + model_dir_str
                    model_dir = Path(model_dir1)
                    if model_dir.exists():
                        with open(str(model_dir), 'r') as foob:
                            content = foob.read()
                            if (a in content) or (b in content) or (c in content) or (d in content) or (
                                    e in content) or (f in content):
                                continue
                            else:
                                # model_name = os.path.basename(model_dir)
                                license_check = 1
                                print('{},The keyword license no exists in the file,please check it!'.format(
                                    model_dir_str))
                            foob.close()
                    else:
                        print('{},The file is not exist!'.format(model_dir_str))
                        license_check = 0
    print('license_check=%d' % license_check)
    if license_check == 1:
        print('License check failed, Please follow the guide to add License:')
        print('https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md')


def get_model_fram(pr_filelist0_str):
    '''
    功能：获取网络框架名称路径
    :param pr_filelist0_dir:
    :return:
    '''
    if '_TensorFlow' in pr_filelist0_str:
        tf_str = '_TensorFlow'
        a = pr_filelist0_str.index(tf_str)
        b = a + 11
        model_fram = pr_filelist0_str[:b]
    elif '_PyTorch' in pr_filelist0_str:
        pt_str = '_PyTorch'
        a = pr_filelist0_str.index(pt_str)
        b = a + 8
        model_fram = pr_filelist0_str[:b]
    elif '_MindSpore' in pr_filelist0_str:
        ms_str = '_MindSpore'
        a = pr_filelist0_str.index(ms_str)
        b = a + 10
        model_fram = pr_filelist0_str[:b]
    else:
        model_fram = ''
    return model_fram


def check_firstlevel_file(path_pr_list, fram_str, modelzoo_dir):
    '''
    :param pr_filelist0_str: 网络路径
    :return:
    功能：检查首层目录是否存在README.md LICENSE文件
    '''
    with open(path_pr_list, 'r') as fooa:
        for filepath_inprlist in fooa:
            filepath_inprlist = filepath_inprlist.strip('\n')
            pr_filelist0_str = filepath_inprlist
            firstlevel_check = 0
            firstlevel_check1 = 0
            firstlevel_check3 = 0
            firstlevel_check4 = 0
            model_fram = get_model_fram(pr_filelist0_str)
            if model_fram == '':  # 网络名称不规范处理
                firstlevel_check4 = 5
            else:  # 网络名称符合规范处理
                # 线上实际网络代码路径
                fram_path1 = fram_str + modelzoo_dir + '/' + get_model_fram(pr_filelist0_str)
                fram_path = fram_path1
                # 判断文件是否存在
                filepath_inprlist = Path(fram_path)
                if filepath_inprlist.exists():
                    # 获取首层目录下所有文件名称
                    # if os.path.isdir(fram_path):
                    # b, c ='README.md', 'LICENSE'
                    a, b, c, d = 'README', 'readme', 'LICENSE', 'Readme'
                    filelist = os.listdir(fram_path)
                    with open('first_filename.txt', 'w') as file:
                        file.write(str(filelist))
                        file.close()
                    with open('first_filename.txt', 'r') as file1:
                        content1 = file1.read()
                        if (a in content1) or (b in content1) or (d in content1):
                            # if (b in filelist) and (c in filelist):
                            firstlevel_check = 0
                        if (a not in content1) and (d not in content1) and (b not in content1):
                            firstlevel_check1 = 2
                            # if b not in filelist:
                            #     firstlevel_check2 = 3
                        if c not in filelist:
                            firstlevel_check3 = 4
                        file1.close()
                else:
                    print('{},The file is not exist!'.format(filepath_inprlist))
                    firstlevel_check = 0
        if firstlevel_check1 == 2:
            print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), a))
            firstlevel_check = 1
        # if firstlevel_check2 == 3:
        #     print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), b))
        #     firstlevel_check = 1
        if firstlevel_check3 == 4:
            print('{},{} is not exist,please check and add it!'.format(get_model_fram(pr_filelist0_str), c))
            firstlevel_check = 1
        if firstlevel_check4 == 5:
            # print('{},The network name of file is not standard，please check name of modelzoo!'.format(pr_filelist0_str))
            firstlevel_check = 0
        print('firstlevel_check=%d' % firstlevel_check)


def funk_file(path_pr_list, fram_str, modelzoo_dir):
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
                # pr_filelist0_str = filepath_inprlist
                funk_file_typr = ['.log', '.pbtxt', '.pb', '.h5', '.so', '.zip', '.tar', '.event', '.tar.gz', '.swp']
                # 获取文件名
                file_name = os.path.basename(str(pr_filelist0_str))
                # 获取文件后缀名
                file_suffix = os.path.splitext(file_name)[1]
                if file_suffix in funk_file_typr:
                    funkfile_check = 1
                    print('{}, The file is Junk file, please check it !'.format(filepath_inprlist))
                if '.ckpt' in str(pr_filelist0_str):
                    funkfile_check = 1
                    print('{}, The file is Junk file, please check it !'.format(filepath_inprlist))
            else:
                funkfile_check = 0
                print('{},The file is not exist!'.format(filepath_inprlist))
        print('funkfile_check=%d' % funkfile_check)


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
            a = 'README'
            b = 'readme'
            c = 'Readme'
            d = 'ReadMe'
            if (a not in file_name) and (b not in file_name) and (c not in file_name) and (d not in file_name):
                pr_filelist0_str = Path(pr_filelist0_str1)
                if pr_filelist0_str.exists():
                    with open(str(pr_filelist0_str), 'r', errors='ignore') as foo:
                        for words in foo:
                            if onelink in words:
                                link = onelink[0:]
                                internal_link_check = 1
                                print('{},This is an internal links that includes {},please check it!'.format(
                                    filepath_inprlist, link))
                            else:
                                continue
                        foo.close()
                else:
                    internal_link_check = 0
                    print('{},The file is not exist!'.format(filepath_inprlist))
        print('internal_link_check=%d' % internal_link_check)


def check_Sensitive_content(path_pr_list, fram_str, modelzoo_dir):
    with open(path_pr_list, 'r') as fooa:
        sensitive_check = 0
        for fram_file_dir in fooa:
            # 去除换行操作
            fram_file_dir = fram_file_dir.strip('\n')
            # pr中文件绝对路径
            pr_filelist0_str1 = fram_str + modelzoo_dir + '/' + fram_file_dir
            # 判断文件是否存在
            pr_filelist0_str = Path(pr_filelist0_str1)
            if pr_filelist0_str.exists():
                # 如果文件存在，打开文件
                with open(str(pr_filelist0_str), 'r', errors='ignore') as foo:
                    for words in foo:
                        if ('0.00' not in words) and ('0.' not in words):
                            if re.findall(r'(00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10})', words) or \
                                    re.findall(r'([ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10})',
                                               words) or \
                                    re.findall(r'([ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10}[ ])',
                                               words) or \
                                    re.findall(r'(.[ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10})',
                                               words) or \
                                    re.findall(r'(.[ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10}.)',
                                               words) or \
                                    re.findall(r'(.[ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10}[ ])',
                                               words) or \
                                    re.findall(
                                        r'(.[ ]00[\d]{5}|00[\d]{6}|00[\d]{7}|00[\d]{8}|00[\d]{9}|00[\d]{10}[ ].)',
                                        words) or \
                                    re.findall(r'(wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11})', words) or \
                                    re.findall(r'([ ]wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11})', words) or \
                                    re.findall(r'([ ]wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11}[ ])', words) or \
                                    re.findall(r'(.[ ]wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11})', words) or \
                                    re.findall(r'(.[ ]wwx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11}.)', words) or \
                                    re.findall(r'(.[ ]wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11}[ ])', words) or \
                                    re.findall(r'(.[ ]wx\d{6}|wx\d{7}|wx\d{8}|wx\d{9}|wx\d{10}|wx\d{11}[ ].)', words):
                                print(
                                    '{}, There may be a job number in the file, please check the line that is: {}'.format(
                                        fram_file_dir, words))
                                sensitive_check = 1
                    foo.close()
                with open(str(pr_filelist0_str), 'r', errors='ignore') as fooc:
                    for words in fooc:
                        if re.findall(
                                r'http://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                                words) or \
                                re.findall(
                                    r'https://\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                                    words):
                            print(
                                '{}, There may be an ip address in the file, please check the line that is:{}'.format(
                                    fram_file_dir, words))
                            sensitive_check = 1
                    fooc.close()
            else:
                sensitive_check = 0
                print('{},The file is not exist!'.format(sensitive_check))
            print('sensitive_check=%d' % sensitive_check)


def main():
    args = init_args()
    path_pr_list = args.pr_filelist_dir
    alink = args.linklisttxt
    # print(path_pr_list)
    tf_str = 'pr_filelist.txt'
    a = path_pr_list.index(tf_str)
    fram_str = path_pr_list[:a]
    modelzoo_dir = 'modelzoo'
    FileSizeLimit = args.FileSizeLimit
    print('=================Start to Check License =================')
    file_scan(path_pr_list, fram_str, modelzoo_dir)
    '''
    :param model_dir : 网络目录
    :param FileSizeLimit : 文件限制大小
    实现功能：扫描文件大小，小于2MB
    '''
    print('=================Start to Check Size of File =================')
    file_size_check(path_pr_list, FileSizeLimit, fram_str, modelzoo_dir, dot=2)
    print('=================Start to Check funk file  =================')
    funk_file(path_pr_list, fram_str, modelzoo_dir)
    # 层级目录检查
    print('=================Start to Check file of First Directory =================')
    # TODO 检查正确的模型根目录
    # check_firstlevel_file(path_pr_list, fram_str, modelzoo_dir)
    print('=================Start to Check Internal Link =================')
    with open(alink, 'r') as food:
        for onelink in food:
            check_link(path_pr_list, fram_str, modelzoo_dir, onelink)
    print('=================Start to Check Sensitive Information =================')
    check_Sensitive_content(path_pr_list, fram_str, modelzoo_dir)


if __name__ == '__main__':
    main()

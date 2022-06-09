# -*- coding: utf-8 -*-
import os,shutil,re
import argparse
import sys

def Get_FileList(file_path):
    """
    # param file_path: the path to the file folder 
    # return: number of LICENSE file, all LICENSE files, all py files, all cpp files
    """
    LICENSE_Count = 0
    LICENSE_Filelist = []
    py_Filelist = []
    cpp_Filelist = []
    # traverse folders, return: current path, current path name, current path files name
    for presentdir, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            # files with path
            file_with_path = os.path.join(presentdir, filename)
            if filename == 'LICENSE' :
                LICENSE_Count = LICENSE_Count + 1
                LICENSE_Filelist.append(file_with_path)
            if file_with_path.split('.')[-1] == 'py':
                py_Filelist.append(file_with_path)
            if file_with_path.split('.')[-1] == 'cpp':
                cpp_Filelist.append(file_with_path)
    return LICENSE_Count, LICENSE_Filelist, py_Filelist, cpp_Filelist

def Read_File(file):
    """
    # param file
    # return: read file by lines
    """
    file_list = []
    # gb18030 encode and ignore errors
    for line in open(file, encoding='gb18030', errors='ignore'):
        file_list.append(line)
    return file_list

def Get_SearchArea(file):
    """
    # param file 
    # return: search area: line0 to end_line (line of the first import)
    """
    file_list = Read_File(file)
    # search line of the first import
    flag_import = 0
    for t in range(len(file_list)):
        # check 'import'
        if re.search("import", file_list[t]):
            flag_import = 1
            import_line = t
            break
    # search area
    if flag_import == 0:
        # not found import, so the search area is total file
        End_line = len(file_list)
    else:
        End_line = import_line
    return End_line

def NEW_File(license_file, old_file, insert_line):
    """
    # param license_file: path to License file
    # param old_file: path to old file
    # param insert_line: insertion position (line)
    """
    # read License file 
    License_list = Read_File(license_file)
    License_list.append("\n")
    # read old file
    old_list = Read_File(old_file)
    # the insert position depend on the insert_line
    # if insert_line=-1 , then insert begins on the top of the file
    if insert_line == -1:
        for k in range(len(old_list)):
            License_list.append(old_list[k])
        # remove old file 
        os.unlink(old_file)
        # new file (the same name to the old file)
        with open(old_file,'w') as F:
            F.writelines(License_list)
            F.close()
    else:
        # new_file text with License
        text_with_License = []
        # add old file : 0 to insert_line
        for k in range(insert_line + 1):
            text_with_License.append(old_list[k])
        # add License
        for k in range(len(License_list)):
            text_with_License.append(License_list[k])
        # add old file : insert_line to end
        for k in range(insert_line + 1, len(old_list)):
            text_with_License.append(old_list[k])
        # remove old file
        os.unlink(old_file)
        # new file (the same name to the old file)
        with open(old_file, 'w') as F:
            F.writelines(text_with_License)
            F.close()
    print(old_file + " : License has been added !")

def LICENSE_File(Model_type, Input_path, Data_path, LICENSE_FileCount, LICENSE_FileList):
    """
    # param Model_type
    # param Input_path
    # param Data_path: some usefull files, like LICENSE file and so on
    # param LICENSE_FileCount: numbers of LICENSE file 
    # param LICENSE_FileList: all LICENSE files
    """
    if Model_type == "TensorFlow":
        file_from = Data_path + "/LICENSE_TF/LICENSE"
    if Model_type == "PyTorch":
        file_from = Data_path + "/LICENSE_Pytorch/LICENSE"

    if LICENSE_FileCount == 0:
        # copy LICENSE file to input path
        shutil.copy(file_from, Input_path)
        print("LICENSE file has been added !")
    else:
        cache_path = Data_path + "/cache"
        cache_path_file = cache_path + "/LICENSE"
        # copy old LICENSE file to cache
        shutil.copy(LICENSE_FileList[0], cache_path)
        # remove all old LICENSE file
        for f in LICENSE_FileList:
            os.unlink(f)
        # copy LICENSE from cache to input path 
        shutil.copy(cache_path_file, Input_path)
        # remove LICENSE file of cache
        os.unlink(cache_path_file)
        print("LICENSE file has been finished ! Excess LICENSE files have been deleted !")

def Add_License_Py(Model_type, Data_path, py_Filelist):
    """
    # param Model_type:
    # param Data_path: some usefull files, like LICENSE file and so on
    # param py_Filelist: all py files
    """
    tf_all = Data_path + "/py_license_tf_all.txt"
    tf_huawei = Data_path + "/py_license_tf_huawei.txt"
    py_all = Data_path + "/py_license_py_all.txt"

    if Model_type == "TensorFlow":
        License_all = tf_all
        License_huawei = tf_huawei
    elif Model_type == "PyTorch":
        License_all = py_all
        License_huawei = py_all
        
    # add License for each file in py_Filelist
    for oldfile in py_Filelist:
        # if file is empty, insert at the top of the file
        if os.path.getsize(oldfile) == 0:
            NEW_File(License_all, oldfile, -1)
        else:
            # read old file
            oldfile_list = Read_File(oldfile)
            # search area
            end_line = Get_SearchArea(oldfile)
            # check 'License' (License_line is the line of the last 'License')
            flag_License = 0
            for t in range(end_line):
                if re.search("License", oldfile_list[t]):
                    flag_License = 1
                    License_line = t
            # not found License then copy all License
            if flag_License == 0:
                # insert position
                # line0 to end_line check 'coding: utf-8' and '#!python'
                flag_utf = 0
                for t in range(end_line):
                    if re.search("utf-8", oldfile_list[t]):
                        flag_utf = 1
                        utf_line = t
                # not found 'coding: utf-8', then check line0 '#!'
                if flag_utf == 0:
                    flag_python = 0
                    if re.search("#!", oldfile_list[0]):
                        flag_python = 1
                    #not found '#!' insert at the top of the file 
                    if flag_python == 0:
                        NEW_File(License_all, oldfile, -1)
                    # found '#!', insert after line0
                    if flag_python == 1:
                        NEW_File(License_all, oldfile, 0)
                # found 'coding: utf-8', insert after line of 'utf-8' (utf_line)
                if flag_utf == 1:
                    NEW_File(License_all, oldfile, utf_line)
            # found License, then search area check 'Huawei'
            if flag_License == 1:
                flag_Huawei = 0
                for t in range(end_line):
                    if re.search("Huawei", oldfile_list[t]):
                        flag_Huawei = 1
                        break
                # not found 'Huawei' insert License_huawei after the line of the last 'License' (License_line)
                if flag_Huawei == 0:
                    NEW_File(License_huawei, oldfile, License_line)
                # found 'Huawei', do nothing
                if flag_Huawei == 1:
                    print(oldfile + " : No need to make changes ! ")
    print("All py files have been processed !")

def Add_License_Cpp(Model_type, Data_path, cpp_Filelist):
    """
    # param Model_type
    # param Data_path: some usefull files, like LICENSE file and so on 
    # param cpp_Filelist: all cpp files
    """
    tf_all = Data_path + "/cpp_license_tf_all.txt"
    tf_huawei = Data_path + "/cpp_license_tf_huawei.txt"
    py_all = Data_path + "/cpp_license_py_all.txt"

    if Model_type == "TensorFlow":
        License_all = tf_all
        License_huawei = tf_huawei
    elif Model_type == "PyTorch":
        License_all = py_all
        License_huawei = py_all

    # add License for each file in cpp_Filelist
    for oldfile in cpp_Filelist:
        # if file is empty, insert at the top of the file
        if os.path.getsize(oldfile) == 0:
            NEW_File(License_all, oldfile, -1)
        else:
            # read old file
            oldfile_list = Read_File(oldfile)
            # search area
            end_line = Get_SearchArea(oldfile)
            # check 'License' (License_line is the line of the last 'License')
            flag_License = 0
            for t in range(end_line):
                if re.search("License", oldfile_list[t]):
                    flag_License = 1
                    License_line = t
            # not found License then copy all License
            if flag_License == 0:
                # insert at the top of the file
                NEW_File(License_all, oldfile, -1)
            # found License, then search area check 'Huawei'
            if flag_License == 1:
                flag_Huawei = 0
                for t in range(end_line):
                    if re.search("Huawei", oldfile_list[t]):
                        flag_Huawei = 1
                        break
                # not found 'Huawei' insert License_huawei after the line of the last 'License' (License_line)
                if flag_Huawei == 0:
                    NEW_File(License_huawei, oldfile, License_line)
                # found 'Huawei', do nothing
                if flag_Huawei == 1:
                    print(oldfile + " : No need to make changes ! ")
    print("All cpp files have been processed !")

def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Check License file and Add License for py/cpp files')
    parser.add_argument('--input_path', type=str,
                        help='Enter the path to the network folder')
    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    args = parse_args(args)
    if args.input_path is None:
        print("Please enter a input_path before running the program !")
        sys.exit()
    elif not re.search("for_TensorFlow", args.input_path) and not re.search("for_PyTorch", args.input_path):
        print("The name is unstandard ! Please use name like **_for_TensorFlow  or  **_for_PyTorch !")
        sys.exit()
    else:
        # get data path (current .py file path)
        data_path = os.path.split(os.path.realpath(__file__))[0] + "/data"
        # get information of the input path
        LICENSE_filecount, LICENSE_filelist, py_filelist, cpp_filelist = Get_FileList(args.input_path)
        # get model type
        if re.search("for_TensorFlow", args.input_path):
            model_type = "TensorFlow"
        elif re.search("for_PyTorch", args.input_path):
            model_type = "PyTorch"
        ''' LICENSE file '''
        LICENSE_File(model_type, args.input_path, data_path, LICENSE_filecount, LICENSE_filelist)
        ''' add License for all py files '''
        if len(py_filelist) == 0:
            print("There has no py files in current input path !")
        else:
            Add_License_Py(model_type, data_path, py_filelist)
        ''' add License for all cpp files '''
        if len(cpp_filelist) == 0:
            print("There has no cpp files in current input path !")
        else:
            Add_License_Cpp(model_type, data_path, cpp_filelist)
        # finished!
        print("Work finished !")


if __name__ == '__main__':
    main()



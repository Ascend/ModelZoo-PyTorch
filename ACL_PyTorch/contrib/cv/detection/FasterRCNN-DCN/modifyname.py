# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
def RenameByOrginBin(aisbinpath):
    source_files = os.listdir(aisbinpath)
    for file in source_files:
        if file == 'sumary.json':
            os.remove(aisbinpath + '/sumary.json')
            print("json removed")
    source_files = os.listdir(aisbinpath)
    for file in source_files:
        oldname = aisbinpath+os.sep+file
        if file[13]=='1':
            filenew=file[:13]+"2.bin"
            newname = aisbinpath+os.sep+filenew
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
    source_files = os.listdir(aisbinpath)
    for file in source_files:
        oldname = aisbinpath+os.sep+file
        if file[13]=='0':
            filenew=file[:13]+"1.bin"
            newname = aisbinpath+os.sep+filenew
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aisbin_path", default="./huaweidata/")
    flags = parser.parse_args()
    RenameByOrginBin(flags.aisbin_path)
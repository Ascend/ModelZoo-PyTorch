# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def main():
    raw_input_en = "./raw_input_200.en-zh.en"
    raw_input_zh = "./raw_input_200.en-zh.zh"
    sort_input_en = "./sort_input_32.en-zh.en"
    sort_input_zh = "./sort_input_32.en-zh.zh"

    connection_data = []
    with open(raw_input_en, 'r') as f1:
        with open(raw_input_zh, 'r') as f2:
            for x, y in zip(f1.readlines(),f2.readlines()):
                connection_data.append([x, y])
        connection_data.sort(key=lambda x: len(x[0].split(' ')))

    data_num = 0
    with open(sort_input_en, 'w') as f1:
        with open(sort_input_zh, 'w') as f2:
            for en, zh in connection_data:
                data_num += 1
                f1.write(en)
                f2.write(zh)
                if data_num == 32:
                    break
    print("sort data saved to: {}; {}.".format(sort_input_en, sort_input_zh))


if __name__ == '__main__':
    main()

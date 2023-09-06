# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import markdown
import argparse
import numpy as np


def get_id_name_link_dict():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
        html = markdown.markdown(content)
    pattern1 = r"""
    <tr>
        <td>\s{,3}\d{6}
        </td><td>\s{,1}
        <a href=\S+>\s{,4}\S+\s{,4}</a>
    """
    matching_pattern1 = re.findall(pattern1, html)
    pattern2 = r"""<p>\S+\d{3,4}\S+</p>"""
    matching_pattern2 = re.findall(pattern2, html)
    number = "".join([i for i in matching_pattern2[0] if i.isdigit()])
    id_name_link_dict = dict()
    for i in matching_pattern1:
        model_link = re.search(r"""href=\S+>""", i).group().replace(">", "").split("=")[1]
        model_id = re.search(r"<td>\s{,3}\d{6}", i).group().split(">")[1]
        model_name = re.search(r">\s{,4}\S+\s{,3}", i.split("<a")[1]).group().replace(" ", "").split(">")[1]
        id_name_link_dict[int(model_id)] = [model_name, model_link]
    return id_name_link_dict, "100" + number


def check_modelid(config):
    id_name_link_dict, num = get_id_name_link_dict()
    model_name = [name[0] for name in id_name_link_dict.values()]
    model_link = [link[1] for link in id_name_link_dict.values()]
    id = max(id_name_link_dict.keys())
    max_id = max(id, int(num))
    name_link_list = []
    for idx, name in enumerate(model_name):
        if config.model.lower() in name.lower() or name.lower() in config.model.lower():
            name_link_list.append(model_link[idx])
    name_link_list = np.array(name_link_list)
    if name_link_list.size == 0:
        print(f"""The ID of this new model is {str(max_id + 1).rjust(6, "0")}""")
    else:
        print(f"{config.model} maybe is already exits, please check the link: \n {name_link_list}")
        while True:
            check_result = input(
                "If this model is different from the models mentioned in the link, please enter true:")
            if check_result.lower() == "true" or check_result == "1":
                print(f"""The ID of this new model is {str(max_id + 1).rjust(6, "0")}""")
                break
            else:
                print("please enter 1 or true")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check the ModelZoo_Pytorch README.MD and get the modelID')
    parser.add_argument('--model', type=str, default="MobileNetV3", help='name of the model')

    opt = parser.parse_args()
    check_modelid(opt)

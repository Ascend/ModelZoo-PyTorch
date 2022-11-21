# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================


/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = allPos[user];

        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int negitem = 0;
            ptr[(user * perUserNum + pair_i) * row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for (int index = 2; index < neg_num + 2; index++)
            {
                do
                {
                    negitem = randint_(item_num);
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_ByUser(std::vector<int> users, int item_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int row = neg_num + 2;
    int col = users.size();
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < users.size(); user_i++)
    {
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        int negitem = 0;

        ptr[user_i * row] = user;
        ptr[user_i * row + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++)
        {
            do
            {
                negitem = randint_(item_num);
            } while (
                find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
            ptr[user_i * row + neg_i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all",
          "user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users",
          "users"_a, "item_num"_a, "allPos"_a, "neg_num"_a);
}
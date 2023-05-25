# Copyright 2023 Huawei Technologies Co., Ltd
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

using JSON
using Serialization
using DelimitedFiles
using Distances
using ArgParse
include("./experiment/utils.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--datapath"
            arg_type=String
            default="./data"
        "--result_dir"
            arg_type=String
            default="./result"
        "--prefix"
            arg_type=String
            default="exp1"
    end
    parse_args(s; as_symbols=true)
end

# get config
datapath = args[:datapath]
result_dir = args[:result_dir]
prefix = args[:prefix]
num_query = 1000
labelfile = joinpath(datapath, "$prefix-trj.label")
vecfile = joinpath(datapath, "$prefix-trj.h5")

# convert result to vectors
result_to_vecs = `python3 t2vec_postprocess.py -data $datapath -result_dir $result_dir`
run(result_to_vecs)

# calculate mean rank
vecs = h5open(vecfile, "r") do f
    read(f["layer3"])
end
label = readdlm(labelfile, Int)

query, db = vecs[:, 1:num_query], vecs[:, num_query+1:end]
queryLabel, dbLabel = label[1:num_query], label[num_query+1:end]
query, db = [query[:, i] for i in 1:size(query, 2)], [db[:, i] for i in 1:size(db, 2)];

dbsizes = [20_000, 40_000, 60_000, 80_000, 100_000]
for dbsize in dbsizes
    ranks = ranksearch(query, queryLabel, db[1:dbsize], dbLabel[1:dbsize], euclidean)
    println("mean rank: $(mean(ranks)) with dbsize: $dbsize")
end
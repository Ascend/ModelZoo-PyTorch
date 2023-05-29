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
        "--save_dir"
            arg_type=String
            default="./prep_data"
        "--prefix"
            arg_type=String
            default="exp1"
    end
    parse_args(s; as_symbols=true)
end

# get config
datapath = args[:datapath]
prefix = args[:prefix]
save_dir = args[:save_dir]
param = JSON.parsefile("./hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

if !isfile("$datapath/$cityname.h5")
    println("Please provide the correct hdf5 file $datapath/$cityname.h5")
    exit(1)
end

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4)

println("Building spatial region with:
        cityname=$(region.name),
        minlon=$(region.minlon),
        minlat=$(region.minlat),
        maxlon=$(region.maxlon),
        maxlat=$(region.maxlat),
        xstep=$(region.xstep),
        ystep=$(region.ystep),
        minfreq=$(region.minfreq)")

paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
    println("Loaded $paramfile into region")
else
    println("Cannot find $paramfile")
end

# create querydb 
do_split = true
start = 1_000_000+20_000
num_query = 1000
num_db = 100_000
querydbfile = joinpath(datapath, "$prefix-querydb.h5")
tfile = joinpath(datapath, "$prefix-trj.t")
labelfile = joinpath(datapath, "$prefix-trj.label")
vecfile = joinpath(datapath, "$prefix-trj.h5")

createQueryDB("$datapath/$cityname.h5", start, num_query, num_db,
              (x, y)->(x, y),
              (x, y)->(x, y);
              do_split=do_split,
              querydbfile=querydbfile)
createTLabel(region, querydbfile; tfile=tfile, labelfile=labelfile)

# save to npy
save_npy = `python3 t2vec_preprocess.py -prefix $prefix -data $datapath -save_dir $save_dir`
run(save_npy)

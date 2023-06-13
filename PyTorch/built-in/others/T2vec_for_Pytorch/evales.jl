using JSON
using Serialization
using DelimitedFiles
using Distances
using ArgParse
include("experiment/utils.jl")

args = 
      let s = ArgParseSettings()
    @add_arg_table s begin
        "--data_path"
            arg_type = String
            default = "./data"
        "--pth_path"
            arg_type = String
            default = "./prep_data"
    end
    parse_args(s; as_symbols=true)
end

println("data: ",args[:datapath])
datapath = args[:datapath]
checkpoint=args[:pthpath]
param = JSON.parsefile("./hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4)

println("Building spatial region with:
        cityname=($(region.name)),minlon=($(region.minlon)),
        minlat=($(region.minlat)), maxlon=($(region.maxlon)),
        maxlat=($(region.maxlat)),xstep=($(region.xstep)),
        ystep=($(region.ystep)),minfreq=($(region.minfreq))")



if !isfile("$datapath/$cityname.h5")
    println("Please provide the correct hdf5 file $datapath/$cityname.h5")
    exit(1)
end


println(region.name)
paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
    println("Loaded $paramfile into region")
else
    println("Cannot find $paramfile")
end




## create querydb
prefix = "exp1"
do_split = true
start = 1_000_000+20_000
num_query = 1000
num_db = 100_000
querydbfile = joinpath(datapath, "$prefix-querydb.h5")
tfile = joinpath(datapath, "$prefix-trj.t")
labelfile = joinpath(datapath, "$prefix-trj.label")
vecfile = joinpath(datapath, "$prefix-trj.h5")

println("cityname  $cityname")
createQueryDB("$datapath/$cityname.h5", start, num_query, num_db,
              (x, y)->(x, y),
              (x, y)->(x, y);
              do_split=do_split,
              querydbfile=querydbfile)

createTLabel(region, querydbfile; tfile=tfile, labelfile=labelfile)


t2vec = `python3 -u t2vec.py -data $datapath -mode 2 -vocab_size 18866 -checkpoint $checkpoint -prefix $prefix`
println(t2vec)
run(t2vec)
cd("experiment")
pwd()
vecs = h5open(vecfile, "r") do f
    read(f["layer3"])
end
label = readdlm(labelfile, Int)

println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
query, db = vecs[:, 1:num_query], vecs[:, num_query+1:end]
queryLabel, dbLabel = label[1:num_query], label[num_query+1:end]
query, db = [query[:, i] for i in 1:size(query, 2)], [db[:, i] for i in 1:size(db, 2)];

# without discriminative loss
dbsizes = [20_000, 40_000, 60_000, 80_000, 100_000]
for dbsize in dbsizes
    ranks = ranksearch(query, queryLabel, db[1:dbsize], dbLabel[1:dbsize], euclidean)
    println("mean rank: $(mean(ranks)) with dbsize: $dbsize")
end

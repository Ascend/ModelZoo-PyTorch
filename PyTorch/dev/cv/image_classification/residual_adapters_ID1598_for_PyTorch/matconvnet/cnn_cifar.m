function [net, info] = cnn_cifar(varargin)

%Demonstrates ResNet (with preactivation) on:
%CIFAR-10 and CIFAR-100 (tested for depth 164)
% cnn_cifar2('modelType', 'new_conv1','GPU', 4, 'batchSize', 128,'momentum', 0.95, 'weightDecay', 0.0001, 'Nclass', 10, 'learningRate', [0.1*ones(1,80) 0.01*ones(1,10) 0.001*ones(1,30)])

run(fullfile(fileparts(mfilename('fullpath')),'matconvnet-1.0-beta25','matlab', 'vl_setupnn.m')) ;

opts.modelType = 'new_conv1' ;
opts.GPU=[];
opts.batchSize=128;
opts.weightDecay=0.0001;
opts.momentum=0.9;
opts.Nclass=10;
opts.filterDepths = [];
opts.learningRate = [0.01*ones(1,3) 0.1*ones(1,80) 0.01*ones(1,10) 0.001*ones(1,20)] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

datas='cifar';
opts.expDir = sprintf('/scratch/shared/nfs1/srebuffi/MCN/%s_%d-%s-D2%d-R%d',datas, opts.Nclass, opts.modelType);

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', datas) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [opts.GPU]; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'reduce_stride'
    net = cnn_resnet_preact_reduce_stride('Nclass', opts.Nclass);
  case 'new_conv1'
    net = cnn_resnet_preact_new_conv1('Nclass', opts.Nclass);  
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

net.meta.trainOpts.learningRate=opts.learningRate; %update lr
net.meta.trainOpts.batchSize = opts.batchSize; %batch size
net.meta.trainOpts.weightDecay = opts.weightDecay; %weight decay
net.meta.trainOpts.momentum = opts.momentum ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate); %update num. ep.

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    if opts.Nclass==10 && strcmp(datas,'cifar')
        imdb = getCifar10Imdb(opts) ;
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb') ;
    else
        imdb = getCifar100Imdb(opts) ;
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb') ;
    end
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    error('The simplenn structure is not supported for the ResNet architecture');
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
images=cropRand(images) ; %random crop for all samples
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'data', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifar10Imdb(opts)
% -------------------------------------------------------------------------
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

%pad the images to crop later
data = padarray(data,[4,4],128,'both');

%remove mean
r = data(:,:,1,set == 1);
g = data(:,:,3,set == 1);
b = data(:,:,3,set == 1);
meanCifar = [mean(r(:)), mean(g(:)), mean(b(:))];
data = bsxfun(@minus, data, reshape(meanCifar,1,1,3));

%divide by std
stdCifar = [std(r(:)), std(g(:)), std(b(:))];
data = bsxfun(@times, data,reshape(1./stdCifar,1,1,3)) ;

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;

% -------------------------------------------------------------------------
function imdb = getCifar100Imdb(opts)
% -------------------------------------------------------------------------
unpackPath = fullfile(opts.dataDir, 'cifar-100-matlab');
files{1} = fullfile(unpackPath, 'train.mat');
files{2} = fullfile(unpackPath, 'test.mat');
%files{3} = fullfile(unpackPath, 'meta.mat');
file_set = uint8([1, 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.fine_labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

%pad the images to crop later
data = padarray(data,[4,4],128,'both');

% remove mean
r = data(:,:,1,set == 1);
g = data(:,:,3,set == 1);
b = data(:,:,3,set == 1);
meanCifar = [mean(r(:)), mean(g(:)), mean(b(:))];
data = bsxfun(@minus, data, reshape(meanCifar,1,1,3));

%divide by std
stdCifar = [std(r(:)), std(g(:)), std(b(:))];
data = bsxfun(@times, data,reshape(1./stdCifar,1,1,3)) ;

clNames = load(fullfile(unpackPath, 'meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.fine_label_names;
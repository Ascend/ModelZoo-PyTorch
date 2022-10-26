# -*- coding: utf-8 -*-
#
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
#
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.resnet12_2 import resnet12
from models.meta_part_inference_mini import ProtoComNet
from models.PredTrainHead import LinearClassifier, LinearRotateHead

from utils import set_gpu, Timer, count_accuracy, check_dir, log
import pickle
import torch.npu
import os
import time
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).npu()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ResNet':
        network = resnet12().npu()
        network = torch.nn.DataParallel(network)
        fea_dim = 512
    else:
        print ("Cannot recognize the network type")
        assert(False)

    propa_head = ProtoComNet(opt=options, in_dim=fea_dim).npu()
        # Choose the classification head
    if opt.use_trainval == 'True':
        n_classes=80
    else:
        n_classes=64
    if options.pre_head == 'LinearNet':
        pre_head = LinearClassifier(in_dim=fea_dim, n_classes=n_classes).npu()
    elif options.pre_head == 'LinearRotateNet':
        pre_head = LinearRotateHead(in_dim=fea_dim, n_classes=n_classes).npu()
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    if options.phase == 'pretrain':
        from models.classification_heads_orgin import ClassificationHead
    else:
        from models.classification_heads import ClassificationHead
    # Choose the classification head
    if options.head == 'CosineNet':
        cls_head = ClassificationHead(base_learner='CosineNet').npu()
    elif options.head == 'FuseCosNet':
        cls_head = ClassificationHead(base_learner='FuseCos').npu()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, propa_head, pre_head, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader, MiniImageNetPC
        # dataset_trainval = MiniImageNet(phase='trainval')
        if options.phase == 'savepart':
            dataset_train = MiniImageNet(phase='train', do_not_use_random_transf=True)
        elif options.phase == 'metainfer':
            dataset_train = MiniImageNetPC(phase='train', shot=options.train_shot)
        else:
            dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, dataset_test, data_loader)

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pre_train(opt, dataset_train, dataset_val, dataset_test, data_loader):
    data_loader_pre = torch.utils.data.DataLoader
    # Dataloader of Gidaris & Komodakis (CVPR 2018)

    if opt.use_trainval == 'True':
        train_way = 80
        dloader_train = data_loader_pre(
            dataset=dataset_trainval,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
    else:
        train_way = 64
        dloader_train = data_loader_pre(
            dataset=dataset_train,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, propa_head, pre_head, cls_head) = get_model(opt)

    print(list(dict(propa_head.named_parameters()).keys()))
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': pre_head.parameters()}], lr=0.1, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 60 else (0.1 if e < 80 else 0.01 if e < 90 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _, _, _, _ = [x.train() for x in (embedding_net, propa_head, pre_head, cls_head)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train), 1):
            start_time = time.time()
            data, labels = [x.npu() for x in batch]

            if opt.pre_head == 'LinearNet' or opt.pre_head == 'CosineNet':
                emb = embedding_net(data)
                logit = pre_head(emb)
                smoothed_one_hot = one_hot(labels.reshape(-1), train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (train_way - 1)

                log_prb = F.log_softmax(logit.reshape(-1, train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                acc = count_accuracy(logit.reshape(-1, train_way), labels.reshape(-1))
            elif opt.pre_head == 'LinearRotateNet' or opt.pre_head == 'DistRotateNet':
                x_ = []
                y_ = []
                a_ = []
                for j in range(data.shape[0]):
                    x90 = data[j].transpose(2, 1).flip(1)
                    x180 = x90.transpose(2, 1).flip(1)
                    x270 = x180.transpose(2, 1).flip(1)
                    x_ += [data[j], x90, x180, x270]
                    y_ += [labels[j] for _ in range(4)]
                    a_ += [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]

                x_ = Variable(torch.stack(x_, 0)).npu()
                y_ = Variable(torch.stack(y_, 0)).npu()
                a_ = Variable(torch.stack(a_, 0)).npu()
                emb = embedding_net(x_)
                # print(emb.shape)
                logit = pre_head(emb, use_cls=True)
                logit_rotate = pre_head(emb, use_cls=False)
                smoothed_one_hot = one_hot(y_.reshape(-1), train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (train_way - 1)

                log_prb = F.log_softmax(logit.reshape(-1, train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
                rloss = F.cross_entropy(input=logit_rotate, target=a_)
                loss = 0.5 * loss + 0.5 * rloss
                acc = count_accuracy(logit.reshape(-1, train_way), y_.reshape(-1))
            else:
                print("Cannot recognize the pre_head type")
                assert (False)


            train_accuracies.append(acc.item())
            train_losses.append(loss.item())
            step_time = time.time() - start_time
            if (i % 10 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {:.3f}\tAccuracy: {:.3f} % ({:.3f} %) \ttime/step(s):{:.4f}'.format(
                    epoch, i, loss.item(), train_acc_avg, acc,step_time))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _, _, _ = [x.eval() for x in (embedding_net, propa_head, pre_head, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(opt.seed)), 1):
            data_support, labels_support, \
            data_query, labels_query, _, _ = [
                x.npu() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()}, \
                       os.path.join(opt.save_path, 'best_pretrain_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} +- {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} +- {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                    'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                   , os.path.join(opt.save_path, 'last_pretrain_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                       , os.path.join(opt.save_path, 'epoch_{}_pretrain.pth'.format(epoch)))

def part_prototype(opt, dataset_train, dataset_val, dataset_test, data_loader):
    data_loader_pre = torch.utils.data.DataLoader
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader_pre(
        dataset=dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, propa_head, pre_head, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_pretrain_model.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()

    embs = []
    for i, batch in enumerate(tqdm(dloader_train), 1):
        data, labels = [x.npu() for x in batch]
        with torch.no_grad():
            emb = embedding_net(data)
        embs.append(emb)
    embs = torch.cat(embs, dim=0)

    with open('./data/mini_imagenet_part_prior_train.pickle', 'rb') as handle:
        part_prior = pickle.load(handle)
    train_class_name_file = './data/mini_imagenet_catname2label_train.pickle'
    with open(train_class_name_file, 'rb') as handle:
        catname2label_train = pickle.load(handle)

    a = 1
    attr_feature = {}
    for attr_id in part_prior['attribute_id_class_dict'].keys():
        if attr_id not in [part_prior['wnids2id'][wnid] for wnid in part_prior['all_wnids']]:
            attr_im_id = []
            for sel_class_id in list(set(part_prior['attribute_id_class_dict'][attr_id])):
                if sel_class_id in [part_prior['wnids2id'][wnid] for wnid in part_prior['wnids_train']]:
                    sel_class = catname2label_train[part_prior['id2wnids'][sel_class_id]]
                    attr_im_id.extend(dataset_train.label2ind[sel_class])
            attr_im = embs[attr_im_id, :]
            mean = torch.mean(attr_im, dim=0).unsqueeze(dim=0)
            std = torch.std(attr_im, dim=0).unsqueeze(dim=0)
            attr_feature[attr_id] = {'mean': mean, 'std':std}

    with open(os.path.join(opt.save_path, "mini_imagenet_metapart_feature.pickle"), 'wb') as handle:
        pickle.dump(attr_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

    class_feature = {}
    for class_id in part_prior['class_attribute_id_dict'].keys():
        if class_id in [part_prior['wnids2id'][wnid] for wnid in part_prior['wnids_train']]:
            sel_class = catname2label_train[part_prior['id2wnids'][class_id]]
            class_im = embs[dataset_train.label2ind[sel_class], :]
            mean = torch.mean(class_im, dim=0).unsqueeze(dim=0)
            std = torch.std(class_im, dim=0).unsqueeze(dim=0)
            class_feature[sel_class] = {'mean': mean, 'std':std}

    with open(os.path.join(opt.save_path, "mini_imagenet_class_feature.pickle"), 'wb') as handle:
        pickle.dump(class_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

def meta_inference(opt, dataset_train, dataset_val, dataset_test, data_loader):
    data_loader_pre = torch.utils.data.DataLoader
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader_pre(
        dataset=dataset_train,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, propa_head, pre_head, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_pretrain_model.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.eval()

    optimizer = torch.optim.SGD([{'params': propa_head.parameters()}], lr=0.1, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 15 else (0.1 if e < 40 else 0.01 if e < 80 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    train_losses = []
    x_entropy = torch.nn.CrossEntropyLoss()
    max_loss = 10e16
    max_val_acc = 0
    max_test_acc = 0
    for epoch in range(0, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        propa_head.train()
        train_accuracies = []
        for i, batch in enumerate(tqdm(dloader_train), 1):
            data, labels = [x.npu() for x in batch]
            nb, ns, nc, nw, nh = data.shape
            with torch.no_grad():
                data = data.reshape(nb*ns, nc, nw, nh)
                emb = embedding_net(data)
                emb = emb.reshape(nb, ns, -1)
                emb = emb.mean(dim=1)
            proto, proto_true = propa_head(emb, labels)
            loss = F.mse_loss(proto, proto_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (i % 10 == 0):
                train_loss_avg = np.mean(np.array(train_losses))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}({})'.format(
                    epoch, i, loss.item(), train_loss_avg))

        # Evaluate on the validation split
        _, _, _, _ = [x.eval() for x in (embedding_net, propa_head, pre_head, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(opt.seed)), 1):
            data_support, labels_support, \
            data_query, labels_query, k_all, _ = [
                x.npu() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(k_all, propa_head, emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, is_scale=True)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()}, \
                       os.path.join(opt.save_path, 'best_pretrain_model_meta_infer_val_{}w_{}s_{}.pth'.format(opt.test_way, opt.val_shot, opt.head)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} 卤 {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} 卤 {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                    'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                   , os.path.join(opt.save_path, 'last_pretrain_epoch_meta_infer.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()} \
                       , os.path.join(opt.save_path, 'epoch_{}_pretrain_meta_infer.pth'.format(epoch)))

def meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 100,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, propa_head, pre_head, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_pretrain_model_meta_infer_val_{}w_{}s_{}.pth'.format(opt.test_way, opt.val_shot, opt.head)))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    propa_head.load_state_dict(saved_models['propa_head'])
    propa_head.eval()

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': propa_head.parameters()},
                                 {'params': cls_head.parameters()}], lr=0.0001, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 15 else (0.1 if e < 25 else 0.01 if e < 30 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(0, opt.num_epoch + 1):
        if epoch != 0:
            # Train on the training split
            lr_scheduler.step()

            # Fetch the current epoch's learning rate
            epoch_learning_rate = 0.1
            for param_group in optimizer.param_groups:
                epoch_learning_rate = param_group['lr']

            log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                epoch, epoch_learning_rate))

            _, _, _ = [x.train() for x in (embedding_net, propa_head, cls_head)]

            train_accuracies = []
            train_losses = []

            for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
                data_support, labels_support, \
                data_query, labels_query, k_all, _ = [
                    x.npu() for x in batch]

                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

                logit_query = cls_head(k_all, propa_head, emb_query, emb_support, labels_support, opt.train_way, opt.train_shot, is_scale=False)

                smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()

                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())

                if (i % 10 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}\tAccuracy: {} % ({} %)'.format(
                        epoch, i, loss.item(), train_acc_avg, acc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on the validation split
        _, _, _ = [x.eval() for x in (embedding_net, propa_head, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(opt.seed)), 1):
            data_support, labels_support, \
            data_query, labels_query, k_all, _ = [
                x.npu() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logit_query = cls_head(k_all, propa_head, emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, is_scale=True)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'propa_head': propa_head.state_dict(),
                        'pre_head': pre_head.state_dict(), 'cls_head': cls_head.state_dict()}, \
                       os.path.join(opt.save_path, 'best_model_meta_val_{}w_{}s_{}.pth'.format(opt.test_way, opt.val_shot, opt.head)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} 卤 {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} 卤 {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))


def meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, propa_head, pre_head, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}w_{}s_{}.pth'.format(opt.test_way, opt.val_shot, opt.head)))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    propa_head.load_state_dict(saved_models['propa_head'])
    propa_head.eval()

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _, _ = [x.eval() for x in (embedding_net, propa_head, cls_head)]
    test_accuracies = []
    test_losses = []

    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 1):
        data_support, labels_support, \
        data_query, labels_query, k_all, _ = [
            x.npu() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query

        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(k_all, propa_head, emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, is_scale=True)

        loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
        acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

        test_accuracies.append(acc.item())
        test_losses.append(loss.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy: {:.2f} 卤 {:.2f} % (Best)' \
        .format(test_loss_avg, test_acc_avg, test_acc_ci95))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=100,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/meta_part_resnet12_mini')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='FuseCosNet',
                            help='choose which classification head to use. FuseCosNet, CosineNet, ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--pre_head', type=str, default='LinearNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--phase', type=str, default='metatest',
                        help='metainfer, pretrain, savepart, metatrain, metatest')
    parser.add_argument('--use_trainval', type=str, default='False',
                        help='frequency of model saving')
    parser.add_argument('--seed', type=int, default=45,
                        help='number of episodes per batch')

    opt = parser.parse_args()
    seed_torch(opt.seed)
    
    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    if opt.phase == 'pretrain':
        pre_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'metatrain':
        meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'metatest':
        meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'savepart':
        part_prototype(opt, dataset_train, dataset_val, dataset_test, data_loader)
    else:
        meta_inference(opt, dataset_train, dataset_val, dataset_test, data_loader)





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
"""Copyright 漏 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

This class is used for training models and is the core of the framework.

With the help of this class, the user of the framework is able to train and
develop models. The framework gets all the relevant objects as an input, and
all the parameters from a YAML file or a dictionary with the parameters, it
instantiates all the relevant helper objects for training the model and does
the training.
"""

from copsolver.frank_wolfe_solver import FrankWolfeSolver
from copsolver.analytical_solver import AnalyticalSolver
from validator import Validator
from paretomanager.pareto_manager_class import ParetoManager
from commondescentvector.multi_objective_cdv import MultiObjectiveCDV
from metric.metric_at_k import MetricAtK
from loss.loss_class import Loss
import torch.nn as nn
from dataloader.mamo_data_handler import MamoDataHandler
import time
import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import yaml
import sys
import logging
import apex
#from apex import amp
try:
    from apex import amp
except ImportError:
    amp = None
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


class Trainer():
    """The trainer class, the core of the framework, used for training models.

    All the needed objects for this class have to be given through the constructor.
    Additionally, the other parameters needed by this trainer have to be supplied
    in a YAML file named 'trainer_params.yaml' or a dictionary containing the parameters.
    For more details about the parameters supplied in this YAML file, please refer to
    'Attributes from the YAML file' section below.

    Attributes:
        data_handler: A MamoDataHandler object which feeds the data set to the trainer.
        model: A torch.nn.Module object which is the model that is being trained.
        losses: A list of Loss objects which represent the losses/objectives that the
            model is trained on.
        validation_metrics: A list of MetricAtK objects which are used to evaluate
            the model while the training and validation process.
        save_to_path: A path to a directory where the trained models from the Pareto
            front will be saved during training.
        device: A variable indicating whether the model is trained on the gpu or on
            the cpu.
        _train_dataloader: A dataloader object used for feeding the data to the trainer.
        pareto_manager: A ParetoManager which is responsible for maintaining a pareto
            front of models and saving these models on permanent storage.
        validator: A Validator object which is used to evaluate the models on multiple
            objective and multiple losses.
        max_empirical_losses: A list of losses (float) which is the approximation of the
            maximum empirical losses the model will have.
        common_descent_vector: A MultiObjectiveCDV, is responsible for combining the multiple
            gradients from the multiple losses into a single gradient.
        optimizer: A pytorch optimizer which is used to train the model.

    Attributes from the YAML file:
        seed: An integer, used to initialize the numpy and pytorch random seeds, default = 42.
        normalize_gradients: A boolean value, indicating whether to normalize the gradients
            while training the model or not, default = True.
        learning_rate: A float value, the learning rate that is given to the pytorch
            optimizer, if the optimizer is not given in the constructor, default = 1e-3.
        batch_size_training: An integer value, representing the batch sizes in which the data is
            fed to the trainer, default = 500.
        shuffle_training: A boolean value, indicating if the training data should be shuffled,
            default = True.
        drop_last_batch_training: A boolean value, indicating to drop the last incomplete batch,
            if the training dataset size is not divisible by the batch size, default = True.
        batch_size_validation: An integer value, representing the batch sizes in which the data is
            fed to the validator, default = 500.
        shuffle_validation: A boolean value, indicating if the validation data should be shuffled,
            default = True.
        drop_last_batch_validation: A boolean value, indicating to drop the last incomplete batch,
            if the validation dataset size is not divisible by the batch size, default = False.
        number_of_epochs: An integer value, indicating for how many epochs should the model
            be trained, default = 50.
        frank_wolfe_max_iter: An integer value, indicating the maximum number of iterations
            to be used by the frank wolfe algorithm in the commonDescentVector object,
            default = 100.
        anneal: A boolean value, indicating if annealing should be used while training the
            model, default = True.
        beta_start: If the anneal is used, this will be the first value of the beta,
            default = 0.
        beta_cap: If the anneal is used, this will be the maximum value of the beta,
            default = 0.3.
        beta_step: If the anneal is used, this is the amount by which to increase the beta
            every batch, default = 0.3/10000.
    """

    def __init__(self, data_handler, model, losses, validation_metrics,
                 save_to_path, params='yaml_files/trainer_params.yaml',
                 optimizer=None):
        """The constructor which initializes a trainer object.

        Arguments:
            data_handler: A MamoDataHandler object which feeds the data set to the trainer.
            model: A torch.nn.Module object which is the model that is being trained.
            losses: A list of Loss objects which represent the losses/objectives that the
                model is trained on.
            validation_metrics: A list of MetricAtK objects which are used to evaluate
                the model while the training and validation process.
            save_to_path: A path to a directory where the trained models from the Pareto
                front will be saved during training.
            params: Path to the yaml file with the trainger parameters, or a dictionary
                containing the parameters.
            optimizer: A pytorch optimizer which is used to train the model, if it is None,
                a default Adam optimizer is created.
        Raises:
            TypeError: If any of the arguments passed are not an instance of the expected
                class or are None, a TypeError will be raised.
            ValueError: If the directory which save_to_path references is not empty, a
                ValueError will be raised.
        """
        logger.info('Trainer: Started with initializing trainer...')
        self._check_input_(data_handler, model, losses,
                           validation_metrics, save_to_path, optimizer)
        self._read_params(params)
        self.data_handler = data_handler
        self.model = model
        self.losses = losses
        logger.info('Trainer: Losses: ')
        logger.info('Trainer: '.join(['%s ' % loss.name for loss in self.losses]))
        self.validation_metrics = validation_metrics
        logger.info('Trainer: Validation metrics: ')
        logger.info('Trainer: '.join(['%s ' % m.get_name() for m in self.validation_metrics]))
        self.save_to_path = save_to_path
        logger.info('Trainer: Saving models to: %s' % self.save_to_path)
        self.optimizer = optimizer
        # set npu if available
        self.device = torch.device(
            'npu' if torch.npu.is_available() else 'cpu')
        #logger.info('Trainer: Training on device: %s' % self.device)
        logger.info('Trainer: Training on device: {}'.format(self.device))
        self._init_objects()
        logger.info('Trainer: Initialization done.')

    def _check_input_(self, data_handler, model, losses, validation_metrics, save_to_path, optimizer):
        """A helper function for the __init__ to check the input of the constructor.
        """
        if not isinstance(data_handler, MamoDataHandler):
            raise TypeError(
                'Please check you are using the right data handler object, or the right order of the attributes!')
        if not isinstance(model, nn.Module):
            raise TypeError(
                'Please check you are using the right model object, or the right order of the attributes!')
        if not hasattr(model, 'initialize_model'):
            raise TypeError(
                'Please check if your models has initialize_model() method defined!')
        # check if losses is None
        if losses is None:
            raise ValueError(
                'The losses are None, please make sure to give valid losses!')
        if not all([isinstance(x, Loss) for x in losses]):
            raise TypeError(
                'Please check you are using the right loss objects, or the right order of the attributes!')
        # check if there are at least two losses
        if len(losses) < 2:
            raise ValueError(
                'Please check you have defined at least two losses,'
                + ' for training with one loss use the Single Objective Loss class!')
        # check if validation metrics is None
        if validation_metrics is None:
            raise ValueError(
                'The validation_metrics are None, please make sure to give valid validation_metrics!')
        if not all([isinstance(x, MetricAtK) for x in validation_metrics]):
            raise TypeError(
                'Please check you are using the right metric objects, or the right order of the attributes!')
        # check if length is at least 1
        if len(validation_metrics) == 0:
            raise ValueError(
                'Please check you have defined at least one validation metric!')
        if not os.path.exists(save_to_path):
            os.mkdir(save_to_path)
        # checking if the save_to_path directory is empty
        if os.listdir(save_to_path):
            raise ValueError(
                'Please make sure that the directory where you want to save the models is empty!')
        # if the optimizer is not None, than has to be pytorch optimizer object
        if optimizer is not None:
            if not isinstance(optimizer, optim.Optimizer):
                raise TypeError(
                    'Please make sure that the optimizer is a pytorch Optimizer object!')

    def _read_params(self, params):
        """A helper function for the __init__ to read the configuration yaml file.
        """
        logger.info('Trainer: Reading trainer parameters.')
        if type(params) is str:
            with open(params, 'r') as stream:
                params = yaml.safe_load(stream)

        self.seed = int(params.get('seed', 42))
        logger.info('Trainer: Random seed: %d' % self.seed)

        self.normalize_gradients = bool(
            params.get('normalize_gradients', True))
        logger.info('Trainer: Normalize gradients: %s' %
                    self.normalize_gradients)

        self.learning_rate = float(params.get('learning_rate', 1e-3))
        logger.info('Trainer: Learning rate: %f' % self.learning_rate)

        self.batch_size_training = int(params.get('batch_size_training', 500))
        logger.info('Trainer: Batch size training: %d' %
                    self.batch_size_training)

        self.shuffle_training = bool(params.get('shuffle_training', True))
        logger.info('Trainer: Shuffle training: %d' %
                    self.shuffle_training)

        self.drop_last_batch_training = bool(
            params.get('drop_last_batch_training', True))
        logger.info('Trainer: Drop last batch training: %d' %
                    self.drop_last_batch_training)

        self.batch_size_validation = int(
            params.get('batch_size_validation', 500))
        logger.info('Trainer: Batch size validation: %d' %
                    self.batch_size_validation)

        self.shuffle_validation = bool(params.get('shuffle_validation', True))
        logger.info('Trainer: Shuffle validation: %d' %
                    self.shuffle_validation)

        self.drop_last_batch_validation = bool(
            params.get('drop_last_batch_validation', False))
        logger.info('Trainer: Drop last batch validation: %d' %
                    self.drop_last_batch_validation)

        self.number_of_epochs = int(params.get('number_of_epochs', 50))
        logger.info('Trainer: Number of epochs: %f' % self.number_of_epochs)

        self.frank_wolfe_max_iter = int(
            params.get('frank_wolfe_max_iter', 100))
        logger.info('Trainer: Frank Wolfe max iterations: %d' %
                    self.frank_wolfe_max_iter)

        self.anneal = bool(params.get('anneal', True))
        logger.info('Trainer: Annealing: %s' % self.anneal)

        if self.anneal and ('beta_start' not in params or 'beta_cap' not in params or 'beta_step' not in params):
            raise ValueError(('Please make sure that if anneal is set to True, '
                              'the beta_start, beta_cap and beta_step are all '
                              'present in the parameters yaml file!'))
        if self.anneal:
            self.beta_start = float(params.get('beta_start', 0))
            logger.info('Trainer: Beta start: %f' % self.beta_start)
            self.beta_cap = float(params.get('beta_cap', 0.3))
            logger.info('Trainer: Beta cap: %f' % self.beta_cap)
            self.beta_step = float(eval(params.get('beta_step', '0.3/10000')))
            logger.info('Trainer: Beta step: %f' % self.beta_step)

    def _init_objects(self):
        """A helper function for the __init__ to initialize different objects.
        """
        logger.info('Trainer: Initializing helper trainer objects.')
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.model.initialize_model()
        self.model.to(self.device)
        self._train_dataloader = self.data_handler.get_train_dataloader(batch_size=self.batch_size_training,
                                                                        shuffle=self.shuffle_training,
                                                                        drop_last=self.drop_last_batch_training,
                                                                        pin_memory=True)
        self.pareto_manager = ParetoManager(PATH=self.save_to_path)
        val_dataloader = self.data_handler.get_validation_dataloader(
            batch_size=self.batch_size_validation, shuffle=self.shuffle_validation,
            drop_last=self.drop_last_batch_validation)
        self.validator = Validator(
            self.model, val_dataloader, self.validation_metrics, self.losses)
        self.max_empirical_losses = None
        if self.normalize_gradients:
            self.max_empirical_losses = self._compute_max_empirical_losses()
            logger.info('Trainer: Max empirical losses: %s' %
                        self.max_empirical_losses)
        copsolver = None
        if len(self.losses) <= 2:
            copsolver = AnalyticalSolver()
        else:
            copsolver = FrankWolfeSolver(max_iter=self.frank_wolfe_max_iter)
        self.common_descent_vector = MultiObjectiveCDV(
            copsolver=copsolver, max_empirical_losses=self.max_empirical_losses,
            normalized=self.normalize_gradients)
        # create default optimizer
        if self.optimizer is None:
            self.optimizer = apex.optimizers.NpuFusedAdam(
                self.model.parameters(), lr=self.learning_rate)

    def _compute_max_empirical_losses(self):
        """A helper function for approximating the maximum empirical loss the model
        could have. It is called by _init_objects function.
        """
        # approximate the max loss empirically
        max_losses = [0] * len(self.losses)
        cnt = 0

        for batch in self._train_dataloader:
            # fetch data
            x = batch[0]
            x = Variable(x).to(self.device)
            y = batch[1]
            y = Variable(y).to(self.device)
            cnt += 1

            # forward pass
            model_output = self.model(x)

            for i, loss in enumerate(self.losses):
                # if annealing is done, the KL divergence is ignored when computing
                # the max empirical loss, therefore the anneal is set to 0
                if self.anneal:
                    L = loss.compute_loss(y, model_output, anneal=0)
                else:
                    L = loss.compute_loss(y, model_output)
                # compute the moving average term
                max_losses[i] = (cnt - 1) / cnt * \
                    max_losses[i] + 1 / cnt * L.item()
        return max_losses

    def _get_gradient_np(self):
        """A helper function for obtaining the gradients of the model in a numpy
        array.

        Before the first backward call, all grad attributes are set to None, and
        that is when the exception is thrown, and the parameters are returned.
        After the first backward pass, the gradient values are available and are
        returned by this function.
        """
        gradient = []
        try:
            for p in self.model.parameters():
                #gradient.append(p.grad.cpu().detach().numpy().ravel())
                gradient.append(p.grad.reshape(-1).to(self.device))
            #return np.concatenate(gradient)
            return torch.cat(gradient)
        except Exception:
            size = 0
            for p in self.model.parameters():
                size += len(p.cpu().detach().numpy().ravel())

            return np.zeros(shape=size)

    def train(self, args):
        """The main method of this class. By calling this method, the traning process
        starts.
        """
        # model training
        logger.info('Trainer: Started training...')

        if self.anneal:
            beta = self.beta_start

        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                              opt_level=args.apex_opt_level,
                                              loss_scale=args.loss_scale_value,
                                              combine_grad=True)

        for epoch in range(self.number_of_epochs):
            # start time for current epoch
            start_time = time.time()

            # statistics
            training_loss = 0
            average_alpha = [0] * len(self.losses)
            cnt = 0

            # set model in train mode
            self.model.train()

            # do training
            for batch in self._train_dataloader:
                t1=time.time()
                # create x
                x = batch[0]
                x = Variable(x).to(self.device,non_blocking=True)
                y = batch[1]
                y = Variable(y).to(self.device,non_blocking=True)

                # anneal beta
                if self.anneal:
                    beta += self.beta_step
                    beta = beta if beta < self.beta_cap else self.beta_cap

                # calculate the gradients
                gradients = []
                for i, loss in enumerate(self.losses):
                    # forward pass
                    model_output = self.model(x)
                    # calculate loss
                    if self.anneal:
                        L = loss.compute_loss(y, model_output, anneal=beta)
                    else:
                        L = loss.compute_loss(y, model_output)
                    # zero gradient
                    self.optimizer.zero_grad()
                    # backward pass

                    if args.apex:
                        with amp.scale_loss(L, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        L.backward()
                    # get gradient for correctness objective
                    gradients.append(self._get_gradient_np())

                # calculate the losses
                losses_computed = []
                # forward pass
                model_output = self.model(x)

                for i, loss in enumerate(self.losses):
                    if self.anneal:
                        L = loss.compute_loss(y, model_output, anneal=beta)
                    else:
                        L = loss.compute_loss(y, model_output)
                    losses_computed.append(L)

                # get the final loss to compute the common descent vector
                final_loss, alphas = self.common_descent_vector.get_descent_vector(
                    losses_computed, gradients)
                # zero gradient
                self.optimizer.zero_grad()
                # backward pass

                if args.apex:
                    with amp.scale_loss(final_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    final_loss.backward()

                # update parameters
                self.optimizer.step()

                # statistics....
                cnt += 1
                # moving average loss
                training_loss = (cnt - 1) / cnt * \
                    training_loss + 1 / cnt * final_loss.item()
                # moving average alpha
                for i, alpha in enumerate(alphas):
                    average_alpha[i] = (cnt - 1) / cnt * \
                        average_alpha[i] + 1 / cnt * alpha

                # time in milliseconds for current batch
                batch_time = (time.time() - start_time) / cnt
                step_time = (time.time()-t1)
                fps =  self.batch_size_training / step_time
                if cnt < 3:
                    print("step_time = {:.4f}".format(step_time), flush=True)
                if cnt > 100:
                    pass

                # log progress
                if cnt % 10 == 0:
                    average_alpha_string = ', '.join(
                        ['%.4f']*len(average_alpha)) % tuple(average_alpha)
                    logger.info('Trainer: Batch: %d/%d, Batch time: %.2fs, Step time: %.2fs, fps: %.3f' %
                                (cnt,
                                 int(np.round(self.data_handler.get_traindata_len() / self.batch_size_training)),
                                 batch_time,
                                 step_time,
                                 fps)
                                + ' Training loss: %.3f, Alphas: [%s]' %
                                (training_loss, average_alpha_string))

            # do validation
            val_metrics, val_objectives = self.validator.evaluate(
                disable_anneal=self.anneal, verbose=False)
            val_loss = self.validator.combine_objectives(
                val_objectives, alphas=average_alpha, max_normalization=self.max_empirical_losses)

            # add the solution to the pareto manager
            self.pareto_manager.add_solution(val_metrics, self.model)

            # calculate epoch time
            epoch_time = time.time() - start_time


            val_metrics_string = ', '.join(
                ['%.4f']*len(val_metrics)) % tuple(val_metrics)
            val_objectives_string = ', '.join(
                ['%.4f']*len(val_objectives)) % tuple(val_objectives)
            logger.info('Trainer: Epoch: %d, Epoch time: %.2fs, Training LOSS: %.3f, Training FPS: %.3f' %
                        (epoch + 1, epoch_time, training_loss, fps)
                        + ' Validation loss: %.3f, Validation metrics: [%s], Validation losses: [%s]' %
                        (val_loss, val_metrics_string, val_objectives_string))
        return val_loss

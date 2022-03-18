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
all the parameters from a YAML file or a dictionary containing the parameters,
it instantiates all the relevant helper objects for training the model and does
the training.
"""

from validator import Validator
from paretomanager.pareto_manager_class import ParetoManager
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
import logging
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


class SingleObjectiveTrainer():
    """The trainer class, the core of the framework, used for training models.

    All the needed objects for this class have to be given through the
    constructor.
    Additionally, the other parameters needed by this trainer have to be
    supplied in a YAML file named 'trainer_params.yaml' or a dictionary
    containing the parameters.
    For more details about the parameters supplied in this YAML file,
    please refer to 'Attributes from the YAML file' section below.

    Attributes:
        data_handler: A MamoDataHandler object which feeds the data set to the trainer.
        model: A torch.nn.Module object which is the model that is being trained.
        loss: A single loss object which represent the loss/objective that the
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
        optimizer: A pytorch optimizer which is used to train the model.

    Attributes from the YAML file:
        seed: An integer, used to initialize the numpy and pytorch random seeds, default = 42.
        learning_rate: A float value, the learning rate that is given to the pytorch
            optimizer, default = 1e-3.
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
        anneal: A boolean value, indicating if annealing should be used while training the
            model, default = True.
        beta_start: If the anneal is used, this will be the first value of the beta,
            default = 0.
        beta_cap: If the anneal is used, this will be the maximum value of the beta,
            default = 3.
        beta_step: If the anneal is used, this is the amount by which to increase the beta
            every batch, default = 3/50.
    """

    def __init__(self, data_handler, model, loss, validation_metrics,
                 save_to_path, params='yaml_files/trainer_params.yaml',
                 optimizer=None):
        """The constructor which initializes a trainer object.

        Arguments:
            data_handler: A MamoDataHandler object which feeds the data set to the trainer.
            model: A torch.nn.Module object which is the model that is being trained.
            loss: A loss object which represent the losse/objective that the model is
                trained on.
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
        self._check_input_(data_handler, model, loss,
                           validation_metrics, save_to_path, optimizer)
        self._read_params(params)
        self.data_handler = data_handler
        self.model = model
        self.loss = loss
        logger.info('Trainer: Loss: %s' % loss.name)
        self.validation_metrics = validation_metrics
        logger.info('Trainer: Validation metrics: ')
        logger.info('Trainer: '.join(['%s ' % m.get_name() for m in self.validation_metrics]))
        self.save_to_path = save_to_path
        logger.info('Trainer: Saving models to: %s' % self.save_to_path)
        self.optimizer = optimizer
        # set npu if available
        self.device = torch.device(
            'npu' if torch.npu.is_available() else 'cpu')
        logger.info('Trainer: Training on device: %s' % self.device)
        self._init_objects()
        logger.info('Trainer: Initialization done.')

    def _check_input_(self, data_handler, model, loss, validation_metrics, save_to_path, optimizer):
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
        # this checks also if loss is None
        if not isinstance(loss, Loss):
            raise TypeError(
                'Please check you are using the right loss object, or the right order of the attributes!')
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
        logger.info('Trainer: Reading yaml trainer parameters.')
        if type(params) is str:
            with open(params, 'r') as stream:
                params = yaml.safe_load(stream)

        self.seed = int(params.get('seed', 42))
        logger.info('Trainer: Random seed: %d' % self.seed)

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

        self.anneal = bool(params.get('anneal', False))
        logger.info('Trainer: Annealing: %s' % self.anneal)

        if self.anneal and ('beta_start' not in params or 'beta_cap' not in params or 'beta_step' not in params):
            raise ValueError(('Please make sure that if anneal is set to True, '
                              'the beta_start, beta_cap and beta_step are all '
                              'present in the parameters yaml file!'))
        if self.anneal:
            self.beta_start = float(params['beta_start'])
            logger.info('Trainer: Beta start: %f' % self.beta_start)
            self.beta_cap = float(params['beta_cap'])
            logger.info('Trainer: Beta cap: %f' % self.beta_cap)
            self.beta_step = float(eval(params['beta_step']))
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
                                                                        drop_last=self.drop_last_batch_training)
        self.pareto_manager = ParetoManager(PATH=self.save_to_path)
        val_dataloader = self.data_handler.get_validation_dataloader(
            batch_size=self.batch_size_validation, shuffle=self.shuffle_validation,
            drop_last=self.drop_last_batch_validation)
        self.validator = Validator(
            self.model, val_dataloader, self.validation_metrics, [self.loss])
        # create default optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """The main method of this class. By calling this method, the traning process
        starts.
        """
        # model training
        logger.info('Trainer: Started training...')

        if self.anneal:
            beta = self.beta_start

        for epoch in range(self.number_of_epochs):
            # start time for current epoch
            start_time = time.time()

            # statistics
            training_loss = 0
            cnt = 0

            # set model in train mode
            self.model.train()

            # do training
            for batch in self._train_dataloader:
                # create x
                x = batch[0]
                x = Variable(x).to(self.device)
                y = batch[1]
                y = Variable(y).to(self.device)

                # anneal beta
                if self.anneal:
                    beta += self.beta_step
                    beta = beta if beta < self.beta_cap else self.beta_cap

                # forward pass
                model_output = self.model(x)
                # calculate loss
                if self.anneal:
                    L = self.loss.compute_loss(
                        y, model_output, anneal=beta)
                else:
                    L = self.loss.compute_loss(y, model_output)
                # zero gradient
                self.optimizer.zero_grad()
                # backward pass
                L.backward()
                # update parameters
                self.optimizer.step()

                # statistics....
                cnt += 1
                # moving average loss
                training_loss = (cnt - 1) / cnt * \
                    training_loss + 1 / cnt * L.item()

                # time in milliseconds for current batch
                batch_time = (time.time() - start_time) / cnt * 1000

                # log progress
                if cnt % 10 == 0:
                    logger.info('Trainer: Batch: %d/%d, Batch time: %.2fms, Training loss: %.3f' %
                                (cnt, int(np.round(
                                          self.data_handler.get_traindata_len()
                                          / self.batch_size_training)),
                                    batch_time, training_loss))

            # do validation
            val_metrics, val_objectives = self.validator.evaluate(
                disable_anneal=self.anneal, verbose=False)

            # add the solution to the pareto manager
            self.pareto_manager.add_solution(val_metrics, self.model)

            # calculate epoch time
            epoch_time = time.time() - start_time

            val_metrics_string = ', '.join(
                ['%.4f']*len(val_metrics)) % tuple(val_metrics)
            val_objectives_string = ', '.join(
                ['%.4f']*len(val_objectives)) % tuple(val_objectives)
            logger.info('Trainer: Epoch: %d, Epoch time: %.2fs, Training loss: %.3f,' %
                        (epoch + 1, epoch_time, training_loss)
                        + ' Validation loss: %.3f, Validation metrics: [%s], Validation losses: [%s]' %
                        (val_objectives[0], val_metrics_string, val_objectives_string))
        return val_objectives[0]

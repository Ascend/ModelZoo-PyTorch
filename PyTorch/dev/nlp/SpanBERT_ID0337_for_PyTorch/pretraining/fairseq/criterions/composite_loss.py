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
from torch import nn

from fairseq import utils, options
from . import FairseqCriterion, register_criterion


@register_criterion('composite_loss')
class CompositeLoss(FairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the composite loss')
        parser.add_argument('--loss-weights', type=str, metavar='EXPR', default=None,
                            help='if set, provides 1 weight per target for each respective loss')

    def __init__(self, args, task):
        super().__init__(args, task)
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion

        assert saved_criterion != args.underlying_criterion

        self.underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        self.weights = options.eval_str_list(args.loss_weights, type=float)

    class FakeModel(nn.Module):
        def __init__(self, model, net_out, target):
            super(CompositeLoss.FakeModel, self).__init__()
            self.model = model
            self.net_out = net_out
            self.target = target

        def forward(self, **unused):
            return self.net_out

        def get_targets(self, *unused):
            return self.target

        @property
        def decoder(self):
            return self.model.decoder

    def forward(self, model, sample, reduce=True):
        net_outputs = model(**sample['net_input'])
        targets = sample['target']

        bsz = targets[0].size(0)
        loss = net_outputs[0][0].new(1 if reduce else bsz).zero_()

        sample_size = 0
        logging_output = {}
        for i, (o, t) in enumerate(zip(net_outputs[0], targets)):
            m = CompositeLoss.FakeModel(model, (o, net_outputs[1]), t)
            l, ss, logging_output = self.underlying_criterion(m, sample, reduce)
            if self.weights is not None:
                l *= self.weights[i]
            loss += l
            sample_size += ss

        loss.div_(len(targets))
        sample_size /= len(targets)

        logging_output['loss'] = utils.item(loss.data) if reduce else loss.data
        return loss, sample_size, logging_output

    def _aggregate_logging_outputs(self, logging_outputs):
        return self.underlying_criterion._aggregate_logging_outputs(logging_outputs)

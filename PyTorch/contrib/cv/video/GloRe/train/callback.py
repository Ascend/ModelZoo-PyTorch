# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Callback function library
"""
import logging

class Callback(object):

    def __init__(self, with_header=False):
        self.with_header = with_header

    def __call__(self):
        raise NotImplementedError("To be implemented")

    def header(self, epoch=None, batch=None, lr=None, **kwargs):
        str_out = ""
        if self.with_header:
            if lr is not None:
                str_out += "lr: {:s} | ".format(("%.6f"%lr).rjust(9, ' '))
            if epoch is not None:
                str_out += "Epoch {:s} ".format(("[%d]"%epoch).ljust(5, ' '))
            if batch is not None:
                str_out += "Batch {:s} ".format(("[%d]"%batch).ljust(6, ' '))
        return str_out
 
class CallbackList(Callback):

    def __init__(self, *args, with_header=True):
        super(CallbackList, self).__init__(with_header=with_header)
        assert all([issubclass(type(x), Callback) for x in args]), \
                "Callback inputs illegal: {}".format(args)
        self.callbacks = [callback for callback in args]

    def __call__(self, epoch=None, batch=None, lr=None, silent=False, **kwargs):
        str_out = self.header(epoch=epoch, batch=batch, lr=lr, **kwargs)

        for callback in self.callbacks:
            str_out += callback(**kwargs, silent=True) + " "

        if not silent:
            logging.info(str_out)
        return str_out   


####################
# CUSTOMIZED CALLBACKS
####################

class SpeedMonitor(Callback):

    def __init__(self, with_header=False):
        super(SpeedMonitor, self).__init__(with_header=with_header)

    def __call__(self, batch_elapse, update_elapse=None, epoch=None, batch=None, silent=False, batch_size=None, **kwargs):        
        str_out = self.header(epoch, batch)

        if batch_elapse is not None:
            batch_freq = 1./batch_elapse
            if update_elapse is not None:
                update_freq = 1./update_elapse
                if batch_size is not None:
                    str_out += "Speed {: >5.1f} (+{: >2.0f}) sample/sec ".format(batch_freq * batch_size, 
                                                                           (update_freq-batch_freq) * batch_size)
                else:
                    str_out += "Speed {: >5.1f} (+{: >2.0f}) iter/sec ".format(batch_freq, update_freq-batch_freq)
            else:
                if batch_size is not None:
                    str_out += "Speed {:.2f} sample/sec ".format(batch_freq * batch_size)
                else:
                    str_out += "Speed {:.2f} iter/sec ".format(batch_freq)

        if not silent:
            logging.info(str_out)
        return str_out

class MetricPrinter(Callback):

    def __init__(self, with_header=False):
        super(MetricPrinter, self).__init__(with_header=with_header)

    def __call__(self, namevals, epoch=None, batch=None, silent=False, **kwargs):
        str_out = self.header(epoch, batch)

        if namevals is not None:
            for i, nameval in enumerate(namevals):
                name, value = nameval[0]
                str_out += "{} = {:.5f}".format(name, value)
                str_out += ", " if i != (len(namevals)-1) else " "

        if not silent:
            logging.info(str_out)
        return str_out


####################
# TESTING CASES
####################

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    # Test each function
    # [1] Callback
    logging.info("- testing base callback class:")
    c = Callback(with_header=True)
    logging.info(c.header(epoch=1, batch=123))
    
    # [2] SpeedMonitor
    logging.info("- testing speedmonitor:")
    s = SpeedMonitor(with_header=True)
    s(batch_elapse=0.3, epoch=10, batch=31)
    s = SpeedMonitor(with_header=False)
    s(batch_elapse=0.3)

    # [3] DictPrinter
    logging.info("- test dict printer")
    d = MetricPrinter(with_header=True)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]], epoch=10, batch=31)
    d = MetricPrinter(with_header=False)
    d(namevals=[[('acc1',0.123)], [("acc5",0.4453232)]])

    # [4] CallbackList
    logging.info("- test callback list")
    c = CallbackList()
    c = CallbackList(SpeedMonitor(), MetricPrinter())
    c(epoch=10, batch=31, batch_elapse=0.3, namevals=[[('acc1',0.123)], [("acc5",0.4453232)]])

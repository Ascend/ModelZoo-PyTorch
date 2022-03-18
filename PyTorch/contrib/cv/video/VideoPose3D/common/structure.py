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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeStats(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, bz=0):
        self.val = val
        self.sum += val
        self.count += bz
        self.avg = self.sum / self.count


def time_format_convert(sec):
    if sec < 60:
        return _process_single_multiple(int(sec),"second")
    elif sec < 3600:
        minutes = int(sec/60)
        seconds = int(sec % 60)
        return _process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")
    elif sec < 3600 * 24:
        hours = int(sec/3600)
        minutes = int((sec - 3600*hours)/60)
        seconds = int(sec%60)
        return _process_single_multiple(hours,"hour")+_process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")
    else:
        days = int(sec/(3600*24))
        left = sec - 3600*24
        hours = int(left / 3600)
        minutes = int((left-3600*hours)/60)
        seconds = int(left % 60)
        return _process_single_multiple(days,"day")+_process_single_multiple(hours,"hour")+_process_single_multiple(minutes,"minute")+_process_single_multiple(seconds,"second")

    
def _process_single_multiple(num, str_):
    if str_ == "second":
        if num == 0 or num == 1:
            return f"{num} "+str_+"."
        else:
            return f"{num} "+str_+"s."
    else:
        if num == 0 or num == 1:
            return f"{num} "+str_+", "
        else:
            return f"{num} "+str_+"s, "


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map
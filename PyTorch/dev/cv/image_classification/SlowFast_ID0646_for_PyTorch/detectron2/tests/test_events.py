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
# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import tempfile
import unittest

from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter


class TestEventWriter(unittest.TestCase):
    def testScalar(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")
            writer = JSONWriter(json_file)
            for k in range(60):
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(l) for l in f]
                self.assertTrue([int(k["key"]) for k in data] == [19, 39, 59])

    def testScalarMismatchedPeriod(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")

            writer = JSONWriter(json_file)
            for k in range(60):
                if k % 17 == 0:  # write in a differnt period
                    storage.put_scalar("key2", k, smoothing_hint=False)
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(l) for l in f]
                self.assertTrue([int(k.get("key2", 0)) for k in data] == [17, 0, 34, 0, 51, 0])
                self.assertTrue([int(k.get("key", 0)) for k in data] == [0, 19, 0, 39, 0, 59])
                self.assertTrue([int(k["iteration"]) for k in data] == [17, 19, 34, 39, 51, 59])

    def testPrintETA(self):
        with EventStorage() as s:
            p1 = CommonMetricPrinter(10)
            p2 = CommonMetricPrinter()

            s.put_scalar("time", 1.0)
            s.step()
            s.put_scalar("time", 1.0)
            s.step()

            with self.assertLogs("detectron2.utils.events") as logs:
                p1.write()
            self.assertIn("eta", logs.output[0])

            with self.assertLogs("detectron2.utils.events") as logs:
                p2.write()
            self.assertNotIn("eta", logs.output[0])

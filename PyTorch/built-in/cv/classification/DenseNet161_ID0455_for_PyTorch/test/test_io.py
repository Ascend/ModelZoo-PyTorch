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
#import os
import contextlib
import tempfile
import torch
import torchvision.datasets.utils as utils
import torchvision.io as io
import unittest
import sys
import warnings

from common_utils import get_tmp_dir

if sys.version_info < (3,):
    from urllib2 import URLError
else:
    from urllib.error import URLError

try:
    import av
    # Do a version test too
    io.video._check_av_available()
except ImportError:
    av = None


def _create_video_frames(num_frames, height, width):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

    return torch.stack(data, 0)


@contextlib.contextmanager
def temp_video(num_frames, height, width, fps, lossless=False, video_codec=None, options=None):
    if lossless:
        assert video_codec is None, "video_codec can't be specified together with lossless"
        assert options is None, "options can't be specified together with lossless"
        video_codec = 'libx264rgb'
        options = {'crf': '0'}

    if video_codec is None:
        video_codec = 'libx264'
    if options is None:
        options = {}

    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data


@unittest.skipIf(av is None, "PyAV unavailable")
class Tester(unittest.TestCase):
    # compression adds artifacts, thus we add a tolerance of
    # 6 in 0-255 range
    TOLERANCE = 6

    def test_write_read_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            lv, _, info = io.read_video(f_name)

            self.assertTrue(data.equal(lv))
            self.assertEqual(info["video_fps"], 5)

    def test_read_timestamps(self):
        with temp_video(10, 300, 300, 5) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)

            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f_name)
            stream = container.streams[0]
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)

    def test_read_partial_video(self):
        with temp_video(10, 300, 300, 5, lossless=True) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            for start in range(5):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue(s_data.equal(lv))

            lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
            self.assertEqual(len(lv), 4)
            self.assertTrue(data[4:8].equal(lv))

    def test_read_partial_video_bframes(self):
        # do not use lossless encoding, to test the presence of B-frames
        options = {'bframes': '16', 'keyint': '10', 'min-keyint': '4'}
        with temp_video(100, 300, 300, 5, options=options) as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)
            for start in range(0, 80, 20):
                for l in range(1, 4):
                    lv, _, _ = io.read_video(f_name, pts[start], pts[start + l - 1])
                    s_data = data[start:(start + l)]
                    self.assertEqual(len(lv), l)
                    self.assertTrue((s_data.float() - lv.float()).abs().max() < self.TOLERANCE)

            lv, _, _ = io.read_video(f_name, pts[4] + 1, pts[7])
            self.assertEqual(len(lv), 4)
            self.assertTrue((data[4:8].float() - lv.float()).abs().max() < self.TOLERANCE)

    def test_read_packed_b_frames_divx_file(self):
        with get_tmp_dir() as temp_dir:
            name = "hmdb51_Turnk_r_Pippi_Michel_cartwheel_f_cm_np2_le_med_6.avi"
            f_name = os.path.join(temp_dir, name)
            url = "https://download.pytorch.org/vision_tests/io/" + name
            try:
                utils.download_url(url, temp_dir)
                pts, fps = io.read_video_timestamps(f_name)
                self.assertEqual(pts, sorted(pts))
                self.assertEqual(fps, 30)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_read_timestamps_from_packet(self):
        with temp_video(10, 300, 300, 5, video_codec='mpeg4') as (f_name, data):
            pts, _ = io.read_video_timestamps(f_name)

            # note: not all formats/codecs provide accurate information for computing the
            # timestamps. For the format that we use here, this information is available,
            # so we use it as a baseline
            container = av.open(f_name)
            stream = container.streams[0]
            # make sure we went through the optimized codepath
            self.assertIn(b'Lavc', stream.codec_context.extradata)
            pts_step = int(round(float(1 / (stream.average_rate * stream.time_base))))
            num_frames = int(round(float(stream.average_rate * stream.time_base * stream.duration)))
            expected_pts = [i * pts_step for i in range(num_frames)]

            self.assertEqual(pts, expected_pts)

    # TODO add tests for audio


if __name__ == '__main__':
    unittest.main()

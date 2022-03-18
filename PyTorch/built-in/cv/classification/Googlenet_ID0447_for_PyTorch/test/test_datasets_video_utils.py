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
#import contextlib
import os
import torch
import unittest

from torchvision import io
from torchvision.datasets.video_utils import VideoClips, unfold

from common_utils import get_tmp_dir


@contextlib.contextmanager
def get_list_of_videos(num_videos=5, sizes=None, fps=None):
    with get_tmp_dir() as tmp_dir:
        names = []
        for i in range(num_videos):
            if sizes is None:
                size = 5 * (i + 1)
            else:
                size = sizes[i]
            if fps is None:
                f = 5
            else:
                f = fps[i]
            data = torch.randint(0, 255, (size, 300, 400, 3), dtype=torch.uint8)
            name = os.path.join(tmp_dir, "{}.mp4".format(i))
            names.append(name)
            io.write_video(name, data, fps=f)

        yield names


class Tester(unittest.TestCase):

    def test_unfold(self):
        a = torch.arange(7)

        r = unfold(a, 3, 3, 1)
        expected = torch.tensor([
            [0, 1, 2],
            [3, 4, 5],
        ])
        self.assertTrue(r.equal(expected))

        r = unfold(a, 3, 2, 1)
        expected = torch.tensor([
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6]
        ])
        self.assertTrue(r.equal(expected))

        r = unfold(a, 3, 2, 2)
        expected = torch.tensor([
            [0, 2, 4],
            [2, 4, 6],
        ])
        self.assertTrue(r.equal(expected))

    @unittest.skipIf(not io.video._av_available(), "this test requires av")
    def test_video_clips(self):
        with get_list_of_videos(num_videos=3) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            self.assertEqual(video_clips.num_clips(), 1 + 2 + 3)
            for i, (v_idx, c_idx) in enumerate([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]):
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

            video_clips = VideoClips(video_list, 6, 6)
            self.assertEqual(video_clips.num_clips(), 0 + 1 + 2)
            for i, (v_idx, c_idx) in enumerate([(1, 0), (2, 0), (2, 1)]):
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

            video_clips = VideoClips(video_list, 6, 1)
            self.assertEqual(video_clips.num_clips(), 0 + (10 - 6 + 1) + (15 - 6 + 1))
            for i, v_idx, c_idx in [(0, 1, 0), (4, 1, 4), (5, 2, 0), (6, 2, 1)]:
                video_idx, clip_idx = video_clips.get_clip_location(i)
                self.assertEqual(video_idx, v_idx)
                self.assertEqual(clip_idx, c_idx)

    @unittest.skip("Moved to reference scripts for now")
    def test_video_sampler(self):
        with get_list_of_videos(num_videos=3, sizes=[25, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = RandomClipSampler(video_clips, 3)  # noqa: F821
            self.assertEqual(len(sampler), 3 * 3)
            indices = torch.tensor(list(iter(sampler)))
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1, 2])))
            self.assertTrue(count.equal(torch.tensor([3, 3, 3])))

    @unittest.skip("Moved to reference scripts for now")
    def test_video_sampler_unequal(self):
        with get_list_of_videos(num_videos=3, sizes=[10, 25, 25]) as video_list:
            video_clips = VideoClips(video_list, 5, 5)
            sampler = RandomClipSampler(video_clips, 3)  # noqa: F821
            self.assertEqual(len(sampler), 2 + 3 + 3)
            indices = list(iter(sampler))
            self.assertIn(0, indices)
            self.assertIn(1, indices)
            # remove elements of the first video, to simplify testing
            indices.remove(0)
            indices.remove(1)
            indices = torch.tensor(indices) - 2
            videos = indices // 5
            v_idxs, count = torch.unique(videos, return_counts=True)
            self.assertTrue(v_idxs.equal(torch.tensor([0, 1])))
            self.assertTrue(count.equal(torch.tensor([3, 3])))

    @unittest.skipIf(not io.video._av_available(), "this test requires av")
    def test_video_clips_custom_fps(self):
        with get_list_of_videos(num_videos=3, sizes=[12, 12, 12], fps=[3, 4, 6]) as video_list:
            num_frames = 4
            for fps in [1, 3, 4, 10]:
                video_clips = VideoClips(video_list, num_frames, num_frames, fps)
                for i in range(video_clips.num_clips()):
                    video, audio, info, video_idx = video_clips.get_clip(i)
                    self.assertEqual(video.shape[0], num_frames)
                    self.assertEqual(info["video_fps"], fps)
                    # TODO add tests checking that the content is right

    def test_compute_clips_for_video(self):
        video_pts = torch.arange(30)
        # case 1: single clip
        num_frames = 13
        orig_fps = 30
        duration = float(len(video_pts)) / orig_fps
        new_fps = 13
        clips, idxs = VideoClips.compute_clips_for_video(video_pts, num_frames, num_frames,
                                                         orig_fps, new_fps)
        resampled_idxs = VideoClips._resample_video_idx(int(duration * new_fps), orig_fps, new_fps)
        self.assertEqual(len(clips), 1)
        self.assertTrue(clips.equal(idxs))
        self.assertTrue(idxs[0].equal(resampled_idxs))

        # case 2: all frames appear only once
        num_frames = 4
        orig_fps = 30
        duration = float(len(video_pts)) / orig_fps
        new_fps = 12
        clips, idxs = VideoClips.compute_clips_for_video(video_pts, num_frames, num_frames,
                                                         orig_fps, new_fps)
        resampled_idxs = VideoClips._resample_video_idx(int(duration * new_fps), orig_fps, new_fps)
        self.assertEqual(len(clips), 3)
        self.assertTrue(clips.equal(idxs))
        self.assertTrue(idxs.flatten().equal(resampled_idxs))


if __name__ == '__main__':
    unittest.main()

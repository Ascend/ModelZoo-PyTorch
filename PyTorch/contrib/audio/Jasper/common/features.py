import math
import random

import librosa
import torch
import torch.nn as nn

from apex import amp
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny, utils


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = utils.normalize(win_sq, norm=norm) ** 2
    win_sq = utils.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann'):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert (filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='constant')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        # print("input_data",input_data)
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
# stft = STFT()
class BaseFeatures(nn.Module):
    """Base class for GPU accelerated audio preprocessing."""
    __constants__ = ["pad_align", "pad_to_max_duration", "max_len"]

    def __init__(self, pad_align, pad_to_max_duration, max_duration,
                 sample_rate, window_size, window_stride, spec_augment=None,
                 cutout_augment=None):
        super(BaseFeatures, self).__init__()

        self.pad_align = pad_align
        self.pad_to_max_duration = pad_to_max_duration
        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride)

        # Calculate maximum sequence length (# frames)
        if pad_to_max_duration:
            self.max_len = 1 + math.ceil(
                (max_duration * sample_rate - self.win_length) / self.hop_length
            )

        if spec_augment is not None:
            self.spec_augment = SpecAugment(**spec_augment)
        else:
            self.spec_augment = None

        if cutout_augment is not None:
            self.cutout_augment = CutoutAugment(**cutout_augment)
        else:
            self.cutout_augment = None

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, audio, audio_lens, optim_level=0):
        dtype = audio.dtype
        audio = audio.float()
        if optim_level == 1:
            with amp.disable_casts():
                feat, feat_lens = self.calculate_features(audio, audio_lens)
        else:
            feat, feat_lens = self.calculate_features(audio, audio_lens)

        feat = self.apply_padding(feat)

        if self.cutout_augment is not None:
            feat = self.cutout_augment(feat)

        if self.spec_augment is not None:
            feat = self.spec_augment(feat)

        feat = feat.to(dtype)
        return feat, feat_lens

    def apply_padding(self, x):
        if self.pad_to_max_duration:
            x_size = max(x.size(-1), self.max_len)
        else:
            x_size = x.size(-1)

        if self.pad_align > 0:
            pad_amt = x_size % self.pad_align
        else:
            pad_amt = 0

        padded_len = x_size + (self.pad_align - pad_amt if pad_amt > 0 else 0)
        return nn.functional.pad(x, (0, padded_len - x.size(-1)))

class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, freq_masks=0, min_freq=0, max_freq=10, time_masks=0,
                 min_time=0, max_time=10):
        super(SpecAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):
            for _ in range(self.freq_masks):
                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = torch.randint(0, max(1, sh[1] - w), size=(1,))
                mask[idx, f0:f0+w] = 1

            for _ in range(self.time_masks):
                w = torch.randint(self.min_time, self.max_time + 1, size=(1,)).item()
                t0 = torch.randint(0, max(1, sh[2] - w), size=(1,))
                mask[idx, :, t0:t0+w] = 1

        return x.masked_fill(mask, 0)


class CutoutAugment(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, masks=0, min_freq=20, max_freq=20, min_time=5, max_time=5):
        super(CutoutAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.masks = masks
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_time = min_time
        self.max_time = max_time

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):
            for i in range(self.masks):

                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                h = torch.randint(self.min_time, self.max_time + 1, size=(1,)).item()

                f0 = int(random.uniform(0, sh[1] - w))
                t0 = int(random.uniform(0, sh[2] - h))

                mask[idx, f0:f0+w, t0:t0+h] = 1

        return x.masked_fill(mask, 0)


@torch.jit.script
def normalize_batch(x, seq_len, normalize_type: str):
#    print ("normalize_batch: x, seq_len, shapes: ", x.shape, seq_len, seq_len.shape)
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                                 device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                                device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :int(seq_len[i])].mean()
            x_std[i] = x[i, :, :int(seq_len[i])].std()
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


@torch.jit.script
def stack_subsample_frames(x, x_lens, stacking: int = 1, subsampling: int = 1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]

    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()

        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:,:,:x_lens.max().item()]

    return x, x_lens


class FilterbankFeatures(BaseFeatures):
    # For JIT, https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length",
                     "log", "frame_splicing", "normalize"]
    # torchscript: "center" removed due to a bug

    def __init__(self, spec_augment=None, cutout_augment=None,
                 sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hamming", normalize="per_feature", n_fft=None,
                 preemph=0.97, n_filt=64, lowfreq=0, highfreq=None, log=True,
                 dither=1e-5, pad_align=8, pad_to_max_duration=False,
                 max_duration=float('inf'), frame_splicing=1):
        super(FilterbankFeatures, self).__init__(
            pad_align=pad_align, pad_to_max_duration=pad_to_max_duration,
            max_duration=max_duration, sample_rate=sample_rate,
            window_size=window_size, window_stride=window_stride,
            spec_augment=spec_augment, cutout_augment=cutout_augment)

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        #TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.n_filt = n_filt
        self.preemph = preemph
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=n_filt,
                                fmin=lowfreq, fmax=highfreq),
            dtype=torch.float).unsqueeze(0)
        # torchscript
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # print("n_fft", self.n_fft)
        # print("hop_length", self.hop_length)
        # print("win_length", self.win_length)
        # print("window", self.window.to(dtype=torch.float))

        # self.stft = STFT(filter_length=512, hop_length=160,win_length=320)
    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

    # do stft
    # TORCHSCRIPT: center removed due to bug
    # def stft(self, x):
    #     return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
    #                       win_length=self.win_length,pad_mode = "constant",
    #                       window=self.window.to(dtype=torch.float))

    def stft(self, x):
        result = []
        for i in range(x.shape[0]):
            tmp = librosa.stft(x[i].numpy(), n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, pad_mode="reflect",
                               window='hann')
            tmp_real = torch.from_numpy(tmp.real).float()
            tmp_imag = torch.from_numpy(tmp.imag).float()
            tmp = torch.stack((tmp_real, tmp_imag), dim=2)
            result.append(tmp)

        return torch.stack((result), dim=0)
    @torch.no_grad()
    def calculate_features(self, x, seq_len):
        dtype = x.dtype
        seq_len = self.get_seq_len(seq_len)
        # dither
        # print(seq_len)
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        # print(x)
        # print(seq_len)
        # x = x.to("cpu")
        # print("inputxsize",x.size())
        # x = x.numpy()
        x  = self.stft(x)
        # x = torch.from_numpy(x)
        # x = x.npu()
            # get power spectrum
        x = x.pow(2).sum(-1)
        # print("x-stft-size:", x.size())
        # print(x.dtype)
        y = self.fb.to(x.dtype)
        # print("fb_ysize:", y.size())
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            raise ValueError('Frame splicing not supported')

        # normalize if required
        x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch,
        # pad to multiple of `pad_align` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype, device=x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), 0)

        # TORCHSCRIPT: Is this del important? It breaks scripting
        # del mask

        return x.to(dtype), seq_len

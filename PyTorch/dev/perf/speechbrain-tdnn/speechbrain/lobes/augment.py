"""
Combinations of processing algorithms to implement common augmentations.

Examples:
 * SpecAugment
 * Environmental corruption (noise, reverberation)

Authors
 * Peter Plantinga 2020
 * Jianyuan Zhong 2020
"""
import os
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.processing.speech_augmentation import (
    SpeedPerturb,
    DropFreq,
    DropChunk,
    AddBabble,
    AddNoise,
    AddReverb,
)
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend

check_torchaudio_backend()

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"


class SpecAugment(torch.nn.Module):
    """An implementation of the SpecAugment algorithm.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    time_warp : bool
        Whether applying time warping.
    time_warp_window : int
        Time warp window.
    time_warp_mode : str
        Interpolation mode for time warping (default "bicubic").
    freq_mask : bool
        Whether applying freq mask.
    freq_mask_width : int or tuple
        Freq mask width range.
    n_freq_mask : int
        Number of freq mask.
    time_mask : bool
        Whether applying time mask.
    time_mask_width : int or tuple
        Time mask width range.
    n_time_mask : int
        Number of time mask.
    replace_with_zero : bool
        If True, replace masked value with 0, else replace masked value with mean of the input tensor.

    Example
    -------
    >>> aug = SpecAugment()
    >>> a = torch.rand([8, 120, 80])
    >>> a = aug(a)
    >>> print(a.shape)
    torch.Size([8, 120, 80])
    """

    def __init__(
        self,
        time_warp=True,
        time_warp_window=5,
        time_warp_mode="bicubic",
        freq_mask=True,
        freq_mask_width=(0, 20),
        n_freq_mask=2,
        time_mask=True,
        time_mask_width=(0, 100),
        n_time_mask=2,
        replace_with_zero=True,
    ):
        super().__init__()
        assert (
            time_warp or freq_mask or time_mask
        ), "at least one of time_warp, time_mask, or freq_mask should be applied"

        self.apply_time_warp = time_warp
        self.time_warp_window = time_warp_window
        self.time_warp_mode = time_warp_mode

        self.freq_mask = freq_mask
        if isinstance(freq_mask_width, int):
            freq_mask_width = (0, freq_mask_width)
        self.freq_mask_width = freq_mask_width
        self.n_freq_mask = n_freq_mask

        self.time_mask = time_mask
        if isinstance(time_mask_width, int):
            time_mask_width = (0, time_mask_width)
        self.time_mask_width = time_mask_width
        self.n_time_mask = n_time_mask

        self.replace_with_zero = replace_with_zero

    def forward(self, x):
        """Takes in input a tensors and returns an augmented one."""
        if self.apply_time_warp:
            x = self.time_warp(x)
        if self.freq_mask:
            x = self.mask_along_axis(x, dim=2)
        if self.time_mask:
            x = self.mask_along_axis(x, dim=1)
        return x

    def time_warp(self, x):
        """Time warping with torch.nn.functional.interpolate"""
        original_size = x.shape
        window = self.time_warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        time = x.shape[2]
        if time - window <= window:
            return x.view(*original_size)

        # compute center and corresponding window
        c = torch.randint(window, time - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1

        left = torch.nn.functional.interpolate(
            x[:, :, :c],
            (w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True,
        )
        right = torch.nn.functional.interpolate(
            x[:, :, c:],
            (time - w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True,
        )

        x[:, :, :w] = left
        x[:, :, w:] = right

        return x.view(*original_size)

    def mask_along_axis(self, x, dim):
        """Mask along time or frequency axis.

        Arguments
        ---------
        x : tensor
            Input tensor.
        dim : int
            Corresponding dimension to mask.
        """
        original_size = x.shape
        if x.dim() == 4:
            x = x.view(-1, x.shape[2], x.shape[3])

        batch, time, fea = x.shape

        if dim == 1:
            D = time
            n_mask = self.n_time_mask
            width_range = self.time_mask_width
        else:
            D = fea
            n_mask = self.n_freq_mask
            width_range = self.freq_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, n_mask), device=x.device
        ).unsqueeze(2)

        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, n_mask), device=x.device
        ).unsqueeze(2)

        # compute masks
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        if self.replace_with_zero:
            val = 0.0
        else:
            with torch.no_grad():
                val = x.mean()

        x = x.masked_fill_(mask, val)
        return x.view(*original_size)


class TimeDomainSpecAugment(torch.nn.Module):
    """A time-domain approximation of the SpecAugment algorithm.

    This augmentation module implements three augmentations in
    the time-domain.

     1. Drop chunks of the audio (zero amplitude or white noise)
     2. Drop frequency bands (with band-drop filters)
     3. Speed peturbation (via resampling to slightly different rate)

    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See ``speechbrain.processing.speech_augmentation.SpeedPerturb``
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, torch.ones(10))
    >>> feats.shape
    torch.Size([10, 12800])
    """

    def __init__(
        self,
        perturb_prob=1.0,
        drop_freq_prob=1.0,
        drop_chunk_prob=1.0,
        speeds=[95, 100, 105],
        sample_rate=16000,
        drop_freq_count_low=0,
        drop_freq_count_high=3,
        drop_chunk_count_low=0,
        drop_chunk_count_high=5,
        drop_chunk_length_low=1000,
        drop_chunk_length_high=2000,
        drop_chunk_noise_factor=0,
    ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, orig_freq=sample_rate, speeds=speeds
        )
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high,
        )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor,
        )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        with torch.no_grad():
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)

        return waveforms


class EnvCorrupt(torch.nn.Module):
    """Environmental Corruptions for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).
    reverb_sample_rate : int
        Sample rate of input audio signals (rirs) used for reverberation.
    noise_sample_rate: int
        Sample rate of input audio signals used for adding noise.
    clean_sample_rate: int
        Sample rate of original (clean) audio signals.

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = EnvCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10))
    """

    def __init__(
        self,
        reverb_prob=1.0,
        babble_prob=1.0,
        noise_prob=1.0,
        openrir_folder=None,
        openrir_max_noise_len=None,
        reverb_csv=None,
        noise_csv=None,
        noise_num_workers=0,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
        rir_scale_factor=1.0,
        reverb_sample_rate=16000,
        noise_sample_rate=16000,
        clean_sample_rate=16000,
    ):
        super().__init__()

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):

            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(
                openrir_folder,
                open_reverb_csv,
                open_noise_csv,
                openrir_max_noise_len,
            )

            # Specify filepath and sample rate if not specified already
            if not reverb_csv:
                reverb_csv = open_reverb_csv
                reverb_sample_rate = 16000

            if not noise_csv:
                noise_csv = open_noise_csv
                noise_sample_rate = 16000

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                replacements={"rir_root": openrir_folder},
                rir_scale_factor=rir_scale_factor,
                reverb_sample_rate=reverb_sample_rate,
                clean_sample_rate=clean_sample_rate,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None and noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                replacements={"rir_root": openrir_folder},
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
                noise_sample_rate=noise_sample_rate,
                clean_sample_rate=clean_sample_rate,
            )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            if hasattr(self, "add_reverb"):
                try:
                    waveforms = self.add_reverb(waveforms, lengths)
                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                waveforms = self.add_noise(waveforms, lengths)

        return waveforms


def _prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
    """Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        download_file(OPENRIR_URL, filepath, unpack=True)
    else:
        download_file(OPENRIR_URL, filepath)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    try:
        # make sure all processing reached here before main preocess create csv_file
        sb.utils.distributed.ddp_barrier()
        if sb.utils.distributed.if_main_process():
            with open(csv_file, "w") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in open(filelist):

                    # Read file for duration/channel info
                    filename = os.path.join(folder, line.split()[-1])
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:
                        # Delete old file
                        os.remove(filename)
                        for i in range(int(duration / max_length)):
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = (
                                filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                f"{ID}_{i}",
                                str((stop - start) / rate),
                                "$rir_root" + new_filename[len(folder) :],
                                ext,
                                "\n",
                            )
                            w.write(",".join(csv_row))
                    else:
                        w.write(
                            ",".join(
                                (
                                    ID,
                                    str(duration),
                                    "$rir_root" + filename[len(folder) :],
                                    ext,
                                    "\n",
                                )
                            )
                        )
    finally:
        sb.utils.distributed.ddp_barrier()

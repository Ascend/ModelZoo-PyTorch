from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
import librosa
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


class StftCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return stft_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def stft(
        speech,
        speech_length,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
):
    bs = speech.size(0)
    if speech.dim() == 3:
        multi_channel = True
        # speech: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
        speech = speech.transpose(1, 2).reshape(-1, speech.size(1))
    else:
        multi_channel = False

    if win_length is None:
        win_length = n_fft

    window_func = getattr(torch, f"{window}_window")
    window = window_func(win_length, dtype=speech.dtype, device=speech.device)

    stft_kwargs = dict(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=center,
        window=window,
    )

    if window is not None:
        # pad the given window to n_fft
        n_pad_left = (n_fft - window.shape[0]) // 2
        n_pad_right = n_fft - window.shape[0] - n_pad_left
        stft_kwargs["window"] = torch. cat(
            [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
        ).numpy()
    else:
        win_length = (
            win_length if win_length is not None else n_fft
        )
        stft_kwargs["window"] = torch.ones(win_length)

    output = []
    # iterate over istances in a batch
    for i, instance in enumerate(speech):
        stft = librosa.stft(speech[i].numpy(), **stft_kwargs)
        output.append(torch.tensor(np.stack([stft.real, stft.imag], -1)))

    output = torch.stack(output, 0)
    output = output.transpose(1, 2)

    if speech_length is not None:
        if center:
            pad = n_fft // 2
            speech_length = speech_length + 2 * pad

        olens = (speech_length - n_fft) // hop_length + 1
        output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
    else:
        olens = None

    return output, olens

def log_mel(
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
):
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    _mel_options = dict(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
    )

    # Note(kamo): The mel matrix of librosa is different from kaldi.
    melmat = librosa.filters.mel(**_mel_options)
    melmat = torch.from_numpy(melmat.T).float()

    mel_feat = torch.matmul(feat, melmat)
    mel_feat = torch.clamp(mel_feat, min=1e-10)

    if log_base is None:
        logmel_feat = mel_feat.log()
    elif log_base == 2.0:
        logmel_feat = mel_feat.log2()
    elif log_base == 10.0:
        logmel_feat = mel_feat.log10()
    else:
        logmel_feat = mel_feat.log() / torch.log(log_base)

    # Zero padding
    if ilens is not None:
        logmel_feat = logmel_feat.masked_fill(
            make_pad_mask(ilens, logmel_feat, 1), 0.0
        )
    else:
        ilens = feat.new_full(
            [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
        )
    return logmel_feat, ilens


def stft_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    stft_speech, stft_speech_length = stft(output['speech'], output['speech_lengths'])
    output_stft = ComplexTensor(stft_speech[..., 0], stft_speech[..., 1])
    output_power = output_stft.real ** 2 + output_stft.imag ** 2
    output_feats, _ = log_mel(output_power, stft_speech_length)

    output['speech'] = output_feats
    output['speech_lengths'] = stft_speech_length
    output = (uttids, output)
    assert check_return_type(output)
    return output


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    assert check_return_type(output)
    return output
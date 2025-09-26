import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # 计算填充大小
    pad_size = int((n_fft - hop_size) / 2)
    
    # 处理填充操作
    if pad_size > 0:
        # 首先确保输入是一维的 [T]
        while y.dim() > 1:
            y = y.squeeze(0)  # 持续压缩直到只剩一维
        
        # 检查输入长度是否足够进行填充
        if y.shape[-1] >= pad_size * 2:
            # 如果输入长度足够，进行正常填充
            y = torch.nn.functional.pad(y.unsqueeze(0), (pad_size, pad_size), mode="reflect")
            y = y.squeeze(0)
        else:
            # 如果输入长度不够，尝试使用较小的填充大小
            adjusted_pad_size = min(pad_size, max(0, y.shape[-1] // 4))
            # 确保调整后的填充大小不会超过输入长度
            if adjusted_pad_size * 2 <= y.shape[-1]:
                y = torch.nn.functional.pad(y.unsqueeze(0), (adjusted_pad_size, adjusted_pad_size), mode="reflect")
                y = y.squeeze(0)
            # 如果调整后的填充大小仍然超过输入长度，则不进行填充

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

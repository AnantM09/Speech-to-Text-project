import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile


settings = {
    "frame_length": 350,
    "step": 150,
    "num_filters": 28,
    "len_mfcc": 13,
    "trim_db": 35,
}

sample_folder = Path(__file__).resolve().parent.parent / "speech_samples"


def frame_rms(audio, frame_length, step):
    audio = np.asarray(audio, dtype=float)
    starts = np.arange(0, len(audio), step)
    ends = np.minimum(starts + frame_length, len(audio))
    squared_sum = np.concatenate(([0], np.cumsum(audio**2)))
    energies = squared_sum[ends] - squared_sum[starts]
    return np.sqrt(np.maximum(energies, 0) / (ends - starts))


def trim_silence(audio, sample_rate, frame_length, step, trim_db=35):
    levels = frame_rms(audio, frame_length, step)
    if len(levels) == 0 or levels.max() == 0:
        raise ValueError("Recording is empty or entirely silent")

    threshold = levels.max() * 10**(-trim_db / 20)
    above_threshold = levels > threshold
    neighbors = np.zeros_like(above_threshold)
    neighbors[1:] |= above_threshold[:-1]
    neighbors[:-1] |= above_threshold[1:]
    active = np.flatnonzero(above_threshold & neighbors)
    if len(active) == 0:
        active = np.flatnonzero(above_threshold)
    margin = round(0.04 * sample_rate)
    start = max(0, active[0] * step - margin)
    end = min(len(audio), active[-1] * step + frame_length + margin)
    return audio[start:end], start, end


def split_audio(audio, frame_length, step):
    last_start = (len(audio) - 1) // step * step
    padding = max(0, last_start + frame_length - len(audio))
    audio = np.pad(np.asarray(audio, dtype=float), (0, padding))
    return np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::step]


@lru_cache(maxsize=16)
def filter_banks(num_filters, sample_rate, frame_length):
    max_mel = 2595 * math.log10(1 + sample_rate / 1400)
    frequencies = []
    for i in range(num_filters + 2):
        mel = i * max_mel / (num_filters + 1)
        frequencies.append(700 * (10**(mel / 2595) - 1))

    filters = []
    for i in range(num_filters):
        left, centre, right = frequencies[i:i + 3]
        current_filter = []
        for k in range(frame_length // 2 + 1):
            frequency = k * sample_rate / frame_length
            if frequency < left or frequency > right:
                value = 0
            elif frequency <= centre:
                value = (frequency - left) / (centre - left)
            else:
                value = (right - frequency) / (right - centre)
            current_filter.append(value)
        filters.append(current_filter)
    return np.array(filters)


def apply_filter(filters, power):
    return np.asarray(power) @ filters.T


@lru_cache(maxsize=16)
def dct_matrix(num_filters, len_mfcc):
    if not 1 <= len_mfcc <= num_filters:
        raise ValueError("MFCC length must be between 1 and the filter count")
    matrix = []
    for j in range(len_mfcc):
        row = []
        for i in range(num_filters):
            row.append(math.cos(math.pi * j * (i + 0.5) / num_filters))
        matrix.append(row)
    return np.array(matrix)


def mfcc(energies, len_mfcc):
    energies = np.asarray(energies)
    matrix = dct_matrix(energies.shape[-1], len_mfcc)
    return np.log(energies + 1e-10) @ matrix.T


def extract_mfcc(audio, sample_rate, frame_length=350, step=150,
                 num_filters=28, len_mfcc=13, trim_db=35):
    if min(frame_length, step, num_filters, sample_rate) <= 0:
        raise ValueError("Frame, step, filter count and sample rate must be positive")
    if np.ndim(audio) != 1:
        raise ValueError("Expected mono audio")
    if trim_db <= 0:
        raise ValueError("trim_db is a positive distance below maximum RMS")

    audio, start, end = trim_silence(
        audio, sample_rate, frame_length, step, trim_db
    )
    filters = filter_banks(num_filters, sample_rate, frame_length)
    frames = split_audio(audio, frame_length, step)
    spectrum = np.fft.rfft(frames, axis=1)
    power = np.abs(spectrum)**2
    return mfcc(apply_filter(filters, power), len_mfcc)


def load_recordings(folder, feature_settings, limit=None):
    files = sorted(folder.glob("*.wav"))
    if limit is not None:
        if limit < 1 or len(files) < limit:
            raise ValueError(f"Need {limit} WAV recordings in {folder}")
        files = files[:limit]
    if not files:
        raise ValueError(f"No WAV recordings in {folder}")

    matrices = []
    for file in files:
        sample_rate, audio = wavfile.read(file)
        matrices.append(extract_mfcc(audio, sample_rate, **feature_settings))
    return files, matrices


if __name__ == "__main__":
    file = sample_folder / "right" / "00b01445_nohash_0.wav"
    sample_rate, audio = wavfile.read(file)
    print(extract_mfcc(audio, sample_rate, **settings))

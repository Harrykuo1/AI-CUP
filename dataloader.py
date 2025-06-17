from __future__ import annotations
import os
import re
import warnings
from typing import List, Optional
import numpy as np
import pandas as pd
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from torch.utils.data import Dataset
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import pywt
    import scipy.signal as sps
except ImportError:
    pywt = None

def _split_by_cut(signal_array: np.ndarray, cut_points: List[int]) -> list[np.ndarray]:
    segments, start_index = [], 0
    for cut_point in cut_points:
        segments.append(signal_array[start_index : cut_point + 1])
        start_index = cut_point + 1
    if start_index < len(signal_array):
        segments.append(signal_array[start_index:])
    return segments

def _resample(segment: np.ndarray, target_length: int) -> np.ndarray:
    if len(segment) == target_length:
        return segment
    resampled_indices = np.linspace(0, len(segment) - 1, target_length)
    resampled_segment = np.stack(
        [
            np.interp(resampled_indices, np.arange(len(segment)), segment[:, i])
            for i in range(segment.shape[1])
        ],
        axis=1,
    )
    return resampled_segment

def _z_norm(segment: np.ndarray) -> np.ndarray:
    return (segment - segment.mean(0, keepdims=True)) / (segment.std(0, keepdims=True) + 1e-6)

def _band_energy(
    segment: np.ndarray, sample_rate: int = 85, bands=((0, 5), (5, 10), (10, 15), (15, 20))
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(len(segment), 1 / sample_rate)
    magnitudes = np.abs(np.fft.rfft(segment, axis=0))
    energy_features = []
    for low_freq, high_freq in bands:
        band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
        mean_energy = magnitudes[band_mask].mean(0) if band_mask.any() else np.zeros(segment.shape[1])
        energy_features.append(mean_energy)
    return np.concatenate(energy_features)

def _swing_stats_60(segment: np.ndarray) -> np.ndarray:
    if segment.size == 0:
        return np.zeros(60, dtype=float)
    first_derivative = np.diff(segment, axis=0, prepend=segment[[0]])
    second_derivative = np.diff(first_derivative, axis=0, prepend=first_derivative[[0]])
    basic_stats = np.concatenate(
        [
            segment.mean(0), segment.std(0),
            first_derivative.mean(0), first_derivative.std(0),
            second_derivative.mean(0), second_derivative.std(0),
        ]
    )
    return np.nan_to_num(np.concatenate([basic_stats, _band_energy(segment)]))

def _wavelet_peak(segment: np.ndarray) -> np.ndarray:
    if segment.size == 0 or pywt is None:
        return np.zeros((segment.shape[1] * 2 + 3,) if segment.ndim > 1 else 3, dtype=float)
    
    coeffs = pywt.wavedec(segment, "db4", level=2, axis=0, mode="periodisation")
    wavelet_features = np.concatenate([c.mean(0) for c in coeffs[1:]])
    
    signal_magnitude = np.linalg.norm(segment, axis=1)
    peaks, _ = sps.find_peaks(signal_magnitude, distance=10)
    
    if len(peaks) >= 3:
        top_peaks = np.sort(signal_magnitude[peaks])[-3:]
    else:
        top_peaks = np.pad(signal_magnitude[peaks], (0, 3 - len(peaks)))
        
    return np.concatenate([wavelet_features, top_peaks])

class TableTennisDataset(Dataset):
    def __init__(
        self,
        meta_csv: str,
        data_dir: str,
        label_cols: Optional[list[str]],
        is_train: bool = True,
        use_extra_features: bool = True,
        seq_len: int = 128,
        use_wavelet: bool = False,
    ):
        super().__init__()
        self.info = pd.read_csv(meta_csv)
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.is_train = is_train
        self.use_extra_features = use_extra_features
        self.seq_len = seq_len
        self.use_wavelet = use_wavelet and (pywt is not None)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        row = self.info.iloc[idx]
        unique_id = str(int(row["unique_id"]))
        data_folder_path = (
            row.get("data_folder")
            if pd.notna(row.get("data_folder")) and row.get("data_folder") != ""
            else self.data_dir
        )
        signal_data = np.loadtxt(os.path.join(data_folder_path, f"{unique_id}.txt"), dtype=float)

        cut_point_numbers = re.findall(r"-?\d+", str(row.get("cut_point", "")))
        cut_points = [int(n) for n in cut_point_numbers] if cut_point_numbers else [len(signal_data) - 1]
        swings = [_z_norm(seg) for seg in _split_by_cut(signal_data, cut_points)]

        extras = None
        if self.use_extra_features:
            statistical_features = [_swing_stats_60(seg) for seg in swings]
            if self.use_wavelet:
                wavelet_peak_features = [_wavelet_peak(seg) for seg in swings]
                statistical_features = [
                    np.concatenate([stat_feat, wavelet_feat])
                    for stat_feat, wavelet_feat in zip(statistical_features, wavelet_peak_features)
                ]
            extras = statistical_features

        labels = None
        if self.label_cols:
            labels = row[self.label_cols].to_numpy(int)

        return swings, extras, labels

def pad_collate(
    batch: list[tuple[list[np.ndarray], Optional[list[np.ndarray]], Optional[np.ndarray]]],
    is_train: bool = True,
):

    max_swings_per_sample = max(len(b[0]) for b in batch)
    max_timesteps_per_swing = max(max(seg.shape[0] for seg in b[0]) for b in batch)

    batch_swing_tensors, batch_masks, batch_extra_features, batch_labels = [], [], [], []

    for swing_list, extra_features_list, label_array in batch:
        if is_train and extra_features_list is not None:
            augmented_swings = []
            for swing_segment in swing_list:
                swing_segment = np.roll(swing_segment, np.random.randint(-5, 6), axis=0)
                swing_segment += 0.005 * np.random.randn(*swing_segment.shape)
                augmented_swings.append(swing_segment)
            swing_list = augmented_swings

        swing_tensors_list = [
            torch.tensor(_resample(s, max_timesteps_per_swing), dtype=torch.float32).permute(1, 0)
            for s in swing_list
        ]
        
        padding_count = max_swings_per_sample - len(swing_tensors_list)
        swing_tensors_list += [torch.zeros_like(swing_tensors_list[0])] * padding_count
        batch_swing_tensors.append(torch.stack(swing_tensors_list))
        batch_masks.append(torch.tensor([1] * len(swing_list) + [0] * padding_count, dtype=torch.bool))

        if extra_features_list is not None:
            extra_feature_dim = extra_features_list[0].shape[0]
            padded_extra_list = extra_features_list + [np.zeros(extra_feature_dim)] * padding_count
            batch_extra_features.append(torch.tensor(np.stack(padded_extra_list), dtype=torch.float32))
        
        if label_array is not None:
            batch_labels.append(torch.tensor(label_array, dtype=torch.long))

    feats = torch.stack(batch_swing_tensors)
    masks = torch.stack(batch_masks)
    extras = torch.stack(batch_extra_features) if batch_extra_features else None
    labels = torch.stack(batch_labels) if batch_labels else torch.empty(0)
    
    return feats, masks, extras, labels

def pad_collate_eval(batch):
    return pad_collate(batch, is_train=False)
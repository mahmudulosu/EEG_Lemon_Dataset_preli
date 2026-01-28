from __future__ import annotations
import numpy as np
import pywt
import pandas as pd

def extract_dwt_features(segment: pd.DataFrame, wavelet: str = "db4", level: int = 6) -> np.ndarray:
    """
    For each channel column:
      coeffs = wavedec(signal, wavelet, level)
      for each coefficient array:
        - std
        - mean
        - RMS
    Returns 1D feature vector.
    """
    feats = []
    for col in segment.columns:
        signal = segment[col].to_numpy(dtype=float)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            feats.extend([np.std(c), np.mean(c), np.sqrt(np.mean(c ** 2))])
    return np.asarray(feats, dtype=float)

def build_feature_matrix(segments: list[pd.DataFrame], wavelet: str = "db4", level: int = 6) -> np.ndarray:
    X = [extract_dwt_features(seg, wavelet=wavelet, level=level) for seg in segments]
    return np.vstack(X)

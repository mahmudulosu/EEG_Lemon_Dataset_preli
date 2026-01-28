import os
import argparse
import numpy as np
import pandas as pd
import pywt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------
# Helpers
# -------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names (remove spaces/dots) to match channel lists reliably."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(r"[.]+", "", regex=True)
    return df

def safe_wavedec(x: np.ndarray, wavelet: str, level: int):
    """Avoid 'level too high' errors by clipping to max allowed level for this signal length."""
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    use_level = min(level, max_level)
    return pywt.wavedec(x, wavelet, level=use_level)

def segment_signal(df: pd.DataFrame, segment_len: int, total_len: int):
    """
    Keep only first total_len samples, then split into non-overlapping segments of segment_len.
    """
    df = df.iloc[:total_len, :]
    segments = []
    for start in range(0, total_len, segment_len):
        end = start + segment_len
        seg = df.iloc[start:end]
        if len(seg) == segment_len:
            segments.append(seg)
    return segments

def extract_dwt_features(segment: pd.DataFrame, wavelet="db4", level=6):
    """
    For each channel: DWT -> for each coeff array -> [mean, var, RMS].
    """
    feats = []
    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)
        coeffs = safe_wavedec(x, wavelet, level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            feats.extend([
                np.mean(c),
                np.var(c),
                np.sqrt(np.mean(c ** 2)),  # RMS
            ])
    return np.asarray(feats, dtype=float)


# -------------------------
# Dataset builder
# -------------------------
def build_dataset(
    data_dir: str,
    sampling_rate: int,
    seconds: int,
    segment_sec: int,
    wavelet: str,
    level: int,
    keep_only: str,
    selected_channels: list[str] | None,
):
    """
    Builds a segment-level dataset for SUBJECT IDENTIFICATION.
    Labels are subject IDs from filename prefix (before first underscore).
    Example: S001_EyeClose.csv -> label "S001"
    """
    total_len = sampling_rate * seconds
    segment_len = sampling_rate * segment_sec

    X_list, y_list = [], []

    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".csv"):
            continue

        # Optional filtering (e.g., only EyeOpen or only EyeClose)
        if keep_only and keep_only.lower() not in fn.lower():
            continue

        fp = os.path.join(data_dir, fn)
        subject_id = fn.split("_")[0]

        df = pd.read_csv(fp)
        df = clean_columns(df)

        # Drop last column if it is Time (your files have Time as last column)
        df = df.iloc[:, :-1]

        # Optional channel subset (e.g., 16 channels)
        if selected_channels is not None:
            available = [ch for ch in selected_channels if ch in df.columns]
            if len(available) == 0:
                raise KeyError(f"No selected channels found in file: {fn}")
            df = df[available]

        segments = segment_signal(df, segment_len=segment_len, total_len=total_len)

        for seg in segments:
            # Convert to numeric safely
            seg = seg.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            feats = extract_dwt_features(seg, wavelet=wavelet, level=level)
            X_list.append(feats)
            y_list.append(subject_id)

    X = np.asarray(X_list, dtype=float)
    y_str = np.asarray(y_list)

    # Sort subject labels for stable mapping
    unique_subjects = np.array(sorted(set(y_str)))
    label_map = {sid: i for i, sid in enumerate(unique_subjects)}
    y = np.array([label_map[s] for s in y_str], dtype=int)

    return X, y, unique_subjects


# -------------------------
# Train + Evaluate
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Subject identification (multi-class) from EEG CSV using DWT + SVM.")
    ap.add_argument("--data_dir", required=True, help="Path to csv_output folder.")
    ap.add_argument("--sampling_rate", type=int, default=160, help="Sampling rate in Hz (default: 160).")
    ap.add_argument("--seconds", type=int, default=60, help="Use first N seconds from each file (default: 60).")
    ap.add_argument("--segment_sec", type=int, default=5, help="Segment length in seconds (default: 5).")
    ap.add_argument("--wavelet", type=str, default="db4")
    ap.add_argument("--level", type=int, default=6, help="Requested DWT level (auto-clipped if too high).")
    ap.add_argument("--percentile", type=int, default=60, help="ANOVA feature selection percentile.")
    ap.add_argument("--keep_only", type=str, default="", help="Filter files by substring (e.g., EyeOpen or EyeClose).")
    ap.add_argument("--use_16ch", action="store_true", help="Use common 16 EEG channels subset.")
    args = ap.parse_args()

    selected_16ch = [
        "Fp1", "Fp2", "F7", "F8", "F3", "F4", "C3", "Cz",
        "C4", "P7", "P3", "Pz", "P4", "T7", "T8", "P8"
    ] if args.use_16ch else None

    X, y, target_names = build_dataset(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate,
        seconds=args.seconds,
        segment_sec=args.segment_sec,
        wavelet=args.wavelet,
        level=args.level,
        keep_only=args.keep_only,
        selected_channels=selected_16ch,
    )

    print("[Data] X shape:", X.shape)
    print("[Data] #classes (subjects):", len(target_names))
    if len(y) > 0:
        binc = np.bincount(y)
        print("[Data] min samples/class:", binc.min(), "| max samples/class:", binc.max())

    # Train/test split (stratified by subject ID)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Feature selection (fit on train only)
    selector = SelectPercentile(score_func=f_classif, percentile=args.percentile)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)

    # Linear SVM (multi-class handled internally by sklearn)
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train_sel, y_train)

    # Predict
    y_pred = model.predict(X_test_sel)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    ))


if __name__ == "__main__":
    main()

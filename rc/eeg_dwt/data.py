from __future__ import annotations
import os
import pandas as pd
import numpy as np

def segment_dataframe(df: pd.DataFrame, segment_length: int) -> list[pd.DataFrame]:
    segments = []
    n = len(df)
    for start in range(0, n, segment_length):
        end = start + segment_length
        if end <= n:
            segments.append(df.iloc[start:end])
    return segments

def load_binary_ec_eo_segments(eeg_dir: str, segment_length: int):
    """
    Binary task label rule:
      - EC if 'EC' appears in filename
      - EO otherwise if 'EO' appears; if neither, skip
    Also returns subject_id = filename prefix before first underscore.
    """
    all_segments = []
    all_labels = []
    subject_ids = []

    for fn in os.listdir(eeg_dir):
        if not fn.endswith(".csv"):
            continue

        label = None
        if "EC" in fn:
            label = 0
        elif "EO" in fn:
            label = 1
        else:
            continue

        subject_id = fn.split("_")[0]
        fp = os.path.join(eeg_dir, fn)
        df = pd.read_csv(fp)

        segs = segment_dataframe(df, segment_length)
        for seg in segs:
            all_segments.append(seg)
            all_labels.append(label)
            subject_ids.append(subject_id)

    return all_segments, np.asarray(all_labels, dtype=int), np.asarray(subject_ids)

def build_pid_to_label_4class(meta_csv: str, valid_ages: list[str]):
    """
    Map participant_id -> class index:
      0: young_M
      1: young_F
      2: old_M
      3: old_F
    Young ages: 20-25, 25-30
    Old ages: 60-65, 65-70, 70-75
    """
    meta = pd.read_csv(meta_csv)

    age_group_map = {
        "20-25": "young",
        "25-30": "young",
        "60-65": "old",
        "65-70": "old",
        "70-75": "old",
    }

    meta["age_str"] = meta["age"].astype(str).str.strip()
    meta["gender_str"] = meta["gender"].astype(str).str.strip().str.upper()
    meta = meta[meta["age_str"].isin(valid_ages) & meta["gender_str"].isin(["M", "F"])]

    pid_to_label = {}
    for _, row in meta.iterrows():
        pid = str(row["participant_id"])
        age_group = age_group_map.get(row["age_str"])
        g = row["gender_str"]

        if age_group == "young" and g == "M":
            lbl = 0
        elif age_group == "young" and g == "F":
            lbl = 1
        elif age_group == "old" and g == "M":
            lbl = 2
        elif age_group == "old" and g == "F":
            lbl = 3
        else:
            continue

        pid_to_label[pid] = lbl

    return pid_to_label

def load_multiclass_segments(eeg_dir: str, segment_length: int, pid_to_label: dict[str, int]):
    """
    Loads EEG segments where subject_id exists in pid_to_label.
    subject_id = filename prefix before first underscore.
    """
    all_segments = []
    all_labels = []
    subject_ids = []

    for fn in os.listdir(eeg_dir):
        if not fn.endswith(".csv"):
            continue

        subject_id = fn.split("_")[0]
        if subject_id not in pid_to_label:
            continue

        label = pid_to_label[subject_id]
        fp = os.path.join(eeg_dir, fn)
        df = pd.read_csv(fp)

        segs = segment_dataframe(df, segment_length)
        for seg in segs:
            all_segments.append(seg)
            all_labels.append(label)
            subject_ids.append(subject_id)

    return all_segments, np.asarray(all_labels, dtype=int), np.asarray(subject_ids)

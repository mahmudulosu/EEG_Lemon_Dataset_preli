import os
import argparse
import numpy as np
import pandas as pd
import pywt

from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ----------------------------
# Metadata: age -> young/old
# ----------------------------
def build_age_mapping(csv_path: str):
    meta = pd.read_csv(csv_path)

    # Adjust these if your column names differ
    pid_col = "participant_id"
    age_col = "age"

    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {
        "20-25": "young",
        "25-30": "young",
        "60-65": "old",
        "65-70": "old",
        "70-75": "old",
    }

    meta[age_col] = meta[age_col].astype(str).str.strip()
    meta[pid_col] = meta[pid_col].astype(str).str.strip()

    meta = meta[meta[age_col].isin(valid_ages)]
    age_mapping = {row[pid_col]: age_group_mapping[row[age_col]] for _, row in meta.iterrows()}
    return age_mapping


# ----------------------------
# EEG utilities
# ----------------------------
def safe_wavedec(x: np.ndarray, wavelet: str, level: int):
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    use_level = min(level, max_level)
    return pywt.wavedec(x, wavelet, level=use_level)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # make numeric-safe
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def align_eo_ec(eo: pd.DataFrame, ec: pd.DataFrame):
    # keep common columns + common length
    common_cols = [c for c in eo.columns if c in ec.columns]
    if len(common_cols) == 0:
        return None, None
    eo = eo[common_cols]
    ec = ec[common_cols]
    n = min(len(eo), len(ec))
    eo = eo.iloc[:n].reset_index(drop=True)
    ec = ec.iloc[:n].reset_index(drop=True)
    return eo, ec

def segment_df(df: pd.DataFrame, seg_len: int):
    segs = []
    for start in range(0, len(df), seg_len):
        end = start + seg_len
        s = df.iloc[start:end]
        if len(s) == seg_len:
            segs.append(s)
    return segs

def dwt_stats_features(df: pd.DataFrame, wavelet="db4", level=5, use_skew_kurt=False):
    """
    For each channel -> DWT -> for each coeff array -> stats.
    Default stats: mean, var, RMS.
    Optionally add skew/kurtosis if you want later.
    """
    feats = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        coeffs = safe_wavedec(x, wavelet, level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            feats.extend([
                float(np.mean(c)),
                float(np.var(c)),
                float(np.sqrt(np.mean(c ** 2))),  # RMS
            ])
    return np.asarray(feats, dtype=float)


# ----------------------------
# Build dataset (2 modes)
# ----------------------------
def build_subject_level_dataset(data_dir, age_mapping, wavelet, level, merge_mode="concat"):
    """
    One sample per subject using both EO and EC files.
    merge_mode:
      - concat: feature(E0) + feature(EC)
      - mean:   features from mean signal ( (EO+EC)/2 )
      - sum:    features from sum signal (EO+EC)
    """
    eeg_by_subject = {}

    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        parts = fn.split("_")
        if len(parts) < 2:
            continue

        subject_id = parts[0]
        condition = parts[1].split(".")[0]  # EO or EC

        if subject_id not in age_mapping:
            continue

        fp = os.path.join(data_dir, fn)
        df = load_csv(fp)

        eeg_by_subject.setdefault(subject_id, {})
        eeg_by_subject[subject_id][condition] = df

    X, y, sids = [], [], []

    for sid, conds in eeg_by_subject.items():
        if "EO" not in conds or "EC" not in conds:
            continue

        eo, ec = align_eo_ec(conds["EO"], conds["EC"])
        if eo is None:
            continue

        if merge_mode == "concat":
            f_eo = dwt_stats_features(eo, wavelet=wavelet, level=level)
            f_ec = dwt_stats_features(ec, wavelet=wavelet, level=level)
            feats = np.concatenate([f_eo, f_ec], axis=0)
        elif merge_mode == "mean":
            merged = (eo + ec) / 2.0
            feats = dwt_stats_features(merged, wavelet=wavelet, level=level)
        elif merge_mode == "sum":
            merged = (eo + ec)
            feats = dwt_stats_features(merged, wavelet=wavelet, level=level)
        else:
            raise ValueError("merge_mode must be one of: concat, mean, sum")

        X.append(feats)
        y.append(0 if age_mapping[sid] == "young" else 1)
        sids.append(sid)

    return np.asarray(X, float), np.asarray(y, int), np.asarray(sids)


def build_segment_level_dataset(data_dir, age_mapping, wavelet, level, sampling_rate=250, seg_sec=2, condition="EO"):
    """
    Many samples per subject by segmenting a chosen condition (EO or EC).
    Evaluation must be subject-wise (GroupKFold / GroupShuffleSplit).
    """
    seg_len = int(seg_sec * sampling_rate)
    X, y, groups = [], [], []

    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        parts = fn.split("_")
        if len(parts) < 2:
            continue

        sid = parts[0]
        cond = parts[1].split(".")[0]  # EO or EC
        if cond != condition:
            continue

        if sid not in age_mapping:
            continue

        fp = os.path.join(data_dir, fn)
        df = load_csv(fp)

        segs = segment_df(df, seg_len=seg_len)
        for seg in segs:
            feats = dwt_stats_features(seg, wavelet=wavelet, level=level)
            X.append(feats)
            y.append(0 if age_mapping[sid] == "young" else 1)
            groups.append(sid)

    return np.asarray(X, float), np.asarray(y, int), np.asarray(groups)


# ----------------------------
# Models
# ----------------------------
def train_eval_svm(X_train, y_train, X_test, y_test, percentile=60):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectPercentile(f_classif, percentile=percentile)),
        ("clf", SVC(kernel="linear", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_pred

def build_dnn(input_dim: int):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_eval_dnn(X_train, y_train, X_test, y_test, percentile=60, epochs=50, batch=32):
    # scale + select (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    selector = SelectPercentile(f_classif, percentile=percentile)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)

    model = build_dnn(X_train_sel.shape[1])
    es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

    model.fit(
        X_train_sel, y_train,
        epochs=epochs,
        batch_size=batch,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )
    y_prob = model.predict(X_test_sel, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Young vs Old classification using Common_Data EEG + metadata CSV.")
    ap.add_argument("--data_dir", required=True, help="Folder containing EEG CSVs (e.g., Common_Data).")
    ap.add_argument("--meta_csv", required=True, help="participants_LSD_andLEMON.csv path.")
    ap.add_argument("--mode", choices=["subject", "segment"], default="subject",
                    help="subject=one sample per subject using EO+EC, segment=many samples per subject using EO or EC.")
    ap.add_argument("--merge_mode", choices=["concat", "mean", "sum"], default="concat",
                    help="Only for subject mode.")
    ap.add_argument("--condition", choices=["EO", "EC"], default="EO",
                    help="Only for segment mode (choose which condition file to segment).")

    ap.add_argument("--sampling_rate", type=int, default=250)
    ap.add_argument("--seg_sec", type=float, default=2.0, help="Segment length in seconds for segment mode.")
    ap.add_argument("--wavelet", type=str, default="db4")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--percentile", type=int, default=60)
    ap.add_argument("--model", choices=["svm", "dnn"], default="dnn")
    ap.add_argument("--cv", choices=["kfold", "holdout"], default="kfold")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    age_mapping = build_age_mapping(args.meta_csv)
    print(f"[META] participants with valid age bins: {len(age_mapping)}")

    if args.mode == "subject":
        X, y, groups = build_subject_level_dataset(
            args.data_dir, age_mapping,
            wavelet=args.wavelet, level=args.level,
            merge_mode=args.merge_mode
        )
        print("[DATA] subject-level samples:", X.shape[0], "| features:", X.shape[1])
        print("[DATA] label dist (young=0, old=1):", np.bincount(y) if len(y) else "empty")

        if args.cv == "holdout":
            # stratified because one sample per subject
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            if args.model == "svm":
                y_pred = train_eval_svm(X_train, y_train, X_test, y_test, percentile=args.percentile)
            else:
                y_pred = train_eval_dnn(X_train, y_train, X_test, y_test,
                                        percentile=args.percentile, epochs=args.epochs, batch=args.batch)

            acc = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {acc:.4f}")
            print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("\nReport:\n", classification_report(y_test, y_pred, target_names=["young", "old"], zero_division=0))

        else:
            skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
            accs = []
            for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
                X_train, y_train = X[tr], y[tr]
                X_test, y_test = X[te], y[te]

                if args.model == "svm":
                    y_pred = train_eval_svm(X_train, y_train, X_test, y_test, percentile=args.percentile)
                else:
                    y_pred = train_eval_dnn(X_train, y_train, X_test, y_test,
                                            percentile=args.percentile, epochs=args.epochs, batch=args.batch)

                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)
                print(f"Fold {fold:02d} acc: {acc:.4f}")

            print(f"\nMean CV Accuracy: {np.mean(accs):.4f}")

    else:
        # segment mode: must split by subject (group-wise)
        X, y, groups = build_segment_level_dataset(
            args.data_dir, age_mapping,
            wavelet=args.wavelet, level=args.level,
            sampling_rate=args.sampling_rate,
            seg_sec=args.seg_sec,
            condition=args.condition
        )
        print("[DATA] segment-level samples:", X.shape[0], "| features:", X.shape[1])
        print("[DATA] subjects:", len(np.unique(groups)))
        print("[DATA] label dist (young=0, old=1):", np.bincount(y) if len(y) else "empty")

        if args.cv == "holdout":
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr, te = next(gss.split(X, y, groups=groups))
            X_train, y_train = X[tr], y[tr]
            X_test, y_test = X[te], y[te]

            if args.model == "svm":
                y_pred = train_eval_svm(X_train, y_train, X_test, y_test, percentile=args.percentile)
            else:
                y_pred = train_eval_dnn(X_train, y_train, X_test, y_test,
                                        percentile=args.percentile, epochs=args.epochs, batch=args.batch)

            acc = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {acc:.4f}")
            print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("\nReport:\n", classification_report(y_test, y_pred, target_names=["young", "old"], zero_division=0))

        else:
            gkf = GroupKFold(n_splits=args.k)
            accs = []
            for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
                X_train, y_train = X[tr], y[tr]
                X_test, y_test = X[te], y[te]

                if len(np.unique(y_test)) < 2:
                    print(f"Fold {fold:02d} skipped (single-class test fold).")
                    continue

                if args.model == "svm":
                    y_pred = train_eval_svm(X_train, y_train, X_test, y_test, percentile=args.percentile)
                else:
                    y_pred = train_eval_dnn(X_train, y_train, X_test, y_test,
                                            percentile=args.percentile, epochs=args.epochs, batch=args.batch)

                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)
                print(f"Fold {fold:02d} acc: {acc:.4f} | test dist: {np.bincount(y_test)}")

            if accs:
                print(f"\nMean CV Accuracy: {np.mean(accs):.4f} (n={len(accs)} folds)")
            else:
                print("\nNo valid folds evaluated.")

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import pywt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

# --------------------
# Config
# --------------------
SAMPLING_RATE = 250
SEGMENT_SEC = 2
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SEC

DATA_FOLDER = r"C:\research\common_data_eeg_ec"   # EC folder (change to EO if needed)
META_CSV = r"C:\research\MRI\participants_LSD_andLEMON.csv"

WAVELET = "db4"
LEVEL = 5
FS_PERCENT = 60
N_SPLITS = 5
RANDOM_STATE = 42


def load_segments(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    segments = []
    for start in range(0, len(df), SEGMENT_LEN):
        seg = df.iloc[start:start + SEGMENT_LEN]
        if len(seg) == SEGMENT_LEN:
            segments.append(seg)
    return segments


def extract_dwt_features(segment: pd.DataFrame):
    feats = []
    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)
        coeffs = pywt.wavedec(x, WAVELET, level=LEVEL)
        for c in coeffs:
            feats.extend([
                float(np.mean(c)),
                float(np.var(c)),
                float(np.sqrt(np.mean(np.asarray(c) ** 2)))  # RMS
            ])
    return np.asarray(feats, dtype=float)


def main():
    meta = pd.read_csv(META_CSV)
    gender_map = {str(r["participant_id"]): str(r["gender"]).upper() for _, r in meta.iterrows()}

    X, y = [], []
    for fn in os.listdir(DATA_FOLDER):
        if not fn.lower().endswith(".csv"):
            continue

        subject_id = fn.split("_")[0]
        if subject_id not in gender_map:
            continue

        for seg in load_segments(os.path.join(DATA_FOLDER, fn)):
            X.append(extract_dwt_features(seg))
            y.append(gender_map[subject_id])

    X = np.asarray(X, dtype=float)
    y = np.asarray([0 if g == "F" else 1 for g in y], dtype=int)  # F=0, M=1

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_acc = []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        selector = SelectPercentile(f_classif, percentile=FS_PERCENT)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        clf = SVC(kernel="linear", random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_acc.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    print("Fold Accuracies:", [f"{a*100:.2f}%" for a in fold_acc])
    print(f"Mean Accuracy: {np.mean(fold_acc)*100:.2f}%")

    print("\nConfusion Matrix:\n", confusion_matrix(y_true_all, y_pred_all))
    print("\nClassification Report:\n",
          classification_report(y_true_all, y_pred_all, target_names=["Female", "Male"], zero_division=0))


if __name__ == "__main__":
    main()

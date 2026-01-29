import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import detrend

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# CONFIG
# =========================
DATA_FOLDER = r"C:\research\common_data_eeg_eo"  # Eye-Open EEG folder
META_CSV = r"C:\research\MRI\participants_LSD_andLEMON.csv"

SAMPLING_RATE = 250
SEGMENT_SECONDS = 5           # <-- change to 2 / 5 / 30 (your experiments)
SEGMENT_LENGTH = SEGMENT_SECONDS * SAMPLING_RATE

WAVELET = "db4"
WPD_LEVEL = 5

FEATURE_PERCENTILE = 60
KFOLDS = 10
RANDOM_STATE = 42

MODEL_TYPE = "dnn"  # "dnn" or "svm"


# =========================
# HELPERS
# =========================
def load_segments(csv_path: str, segment_length: int):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    segments = []
    for start in range(0, len(df), segment_length):
        seg = df.iloc[start:start + segment_length]
        if len(seg) == segment_length:
            segments.append(seg.to_numpy(dtype=float))
    return segments


def remove_artifacts_and_normalize(segment_np: np.ndarray):
    """
    - Detrend removes linear drift (baseline / slow trend)
    - Z-score normalization per channel
    """
    seg = detrend(segment_np, axis=0)
    seg = (seg - np.mean(seg, axis=0)) / (np.std(seg, axis=0) + 1e-8)
    return seg


def extract_wpd_features(segment_np: np.ndarray):
    """
    Wavelet Packet Decomposition at level=5.
    For each channel -> for each node at level 5:
      mean, RMS
    """
    feats = []
    n_channels = segment_np.shape[1]

    for ch in range(n_channels):
        x = segment_np[:, ch]
        packet = pywt.WaveletPacket(data=x, wavelet=WAVELET, mode="symmetric", maxlevel=WPD_LEVEL)

        for node in packet.get_level(WPD_LEVEL, order="natural"):
            node_data = np.asarray(node.data, dtype=float)
            feats.extend([
                float(np.mean(node_data)),
                float(np.sqrt(np.mean(node_data ** 2)))  # RMS
            ])
    return np.asarray(feats, dtype=float)


def build_dnn(input_dim: int):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =========================
# MAIN PIPELINE
# =========================
def main():
    # ---- Load metadata and create age mapping ----
    meta = pd.read_csv(META_CSV)

    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {
        "20-25": "young", "25-30": "young",
        "60-65": "old", "65-70": "old", "70-75": "old"
    }

    meta = meta[meta["age"].isin(valid_ages)]
    age_map = {row["participant_id"]: age_group_mapping[row["age"]] for _, row in meta.iterrows()}

    # ---- Build dataset ----
    X, y, subject_ids = [], [], []

    for fn in os.listdir(DATA_FOLDER):
        if not fn.lower().endswith(".csv"):
            continue

        subject_id = fn.split("_")[0]
        if subject_id not in age_map:
            continue

        csv_path = os.path.join(DATA_FOLDER, fn)
        segments = load_segments(csv_path, SEGMENT_LENGTH)

        for seg in segments:
            seg = remove_artifacts_and_normalize(seg)
            feats = extract_wpd_features(seg)

            X.append(feats)
            y.append(0 if age_map[subject_id] == "young" else 1)
            subject_ids.append(subject_id)

    X = np.asarray(X)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)

    print("X shape:", X.shape)
    print("y distribution (young=0, old=1):", np.bincount(y))

    if X.size == 0 or len(np.unique(y)) < 2:
        raise ValueError("Dataset empty or only one class found. Check file naming + age mapping.")

    # ---- Subject-wise CV ----
    unique_subjects = np.unique(subject_ids)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold, (train_sub_idx, test_sub_idx) in enumerate(kf.split(unique_subjects), start=1):
        train_subjects = unique_subjects[train_sub_idx]
        test_subjects = unique_subjects[test_sub_idx]

        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature selection (fit only on train fold)
        selector = SelectPercentile(score_func=f_classif, percentile=FEATURE_PERCENTILE)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # ---- Train model ----
        if MODEL_TYPE.lower() == "svm":
            model = SVC(kernel="linear", random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)

        elif MODEL_TYPE.lower() == "dnn":
            model = build_dnn(X_train.shape[1])
            es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                callbacks=[es],
                verbose=0
            )
            y_hat = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()

        else:
            raise ValueError("MODEL_TYPE must be 'dnn' or 'svm'")

        acc = accuracy_score(y_test, y_hat)
        fold_accuracies.append(acc)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_hat.tolist())

        print(f"Fold {fold}/{KFOLDS} Accuracy: {acc*100:.2f}%")

    # ---- Final report ----
    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    print("\n==============================")
    print("Subject-wise CV Results")
    print("==============================")
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Segment length: {SEGMENT_SECONDS}s")
    print(f"Mean Accuracy: {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_pred))

    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, target_names=["young", "old"]))


if __name__ == "__main__":
    main()

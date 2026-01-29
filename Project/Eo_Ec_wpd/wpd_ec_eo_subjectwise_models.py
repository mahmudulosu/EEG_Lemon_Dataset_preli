import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# CONFIG
# =========================
DATA_DIR = r"C:\research\EEG_Domain\combined\combined_new"

SAMPLING_RATE = 250
SEGMENT_SECONDS = 10
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SECONDS

WAVELET = "db4"
WPD_LEVEL = 5

FEATURE_PERCENTILE = 60
KFOLDS = 10
RANDOM_STATE = 42


# =========================
# DATA LOADING + SEGMENTATION
# =========================
def load_eeg_segments(csv_path: str):
    """
    Load EEG from CSV, infer label from filename (EC/EO),
    and segment into fixed-length windows.
    """
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    label = "EC" if "EC" in os.path.basename(csv_path) else "EO"

    segments = []
    for start in range(0, len(df), SEGMENT_LEN):
        seg = df.iloc[start:start + SEGMENT_LEN]
        if len(seg) == SEGMENT_LEN:
            segments.append(seg)

    return segments, label


# =========================
# WPD FEATURE EXTRACTION
# =========================
def extract_wpd_features(segment: pd.DataFrame) -> np.ndarray:
    """
    Wavelet Packet Decomposition (WPD) features.
    For each channel at level WPD_LEVEL:
      mean, var, RMS, kurtosis, skew
    """
    feats = []

    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)

        packet = pywt.WaveletPacket(
            data=x, wavelet=WAVELET, mode="symmetric", maxlevel=WPD_LEVEL
        )

        nodes = packet.get_level(WPD_LEVEL, order="natural")
        for node in nodes:
            nd = np.asarray(node.data, dtype=float)
            feats.extend([
                float(np.mean(nd)),
                float(np.var(nd)),
                float(np.sqrt(np.mean(nd**2))),  # RMS
                float(kurtosis(nd)),
                float(skew(nd))
            ])

    return np.asarray(feats, dtype=float)


# =========================
# MODEL: DNN
# =========================
def build_dnn(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =========================
# SUBJECT-WISE DATASET CREATION
# =========================
def build_subjectwise_dataset(data_dir: str):
    """
    Randomly select ONE segment per subject per condition (EC and EO),
    then extract WPD features.
    """
    subject_segments = {}

    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(data_dir, fn)
        subject_id = fn.split("_")[0]
        segments, label = load_eeg_segments(csv_path)

        if subject_id not in subject_segments:
            subject_segments[subject_id] = {"EC": [], "EO": []}
        subject_segments[subject_id][label].extend(segments)

    np.random.seed(RANDOM_STATE)

    X, y, subjects = [], [], []
    for subject_id, seg_dict in subject_segments.items():
        for condition, segs in seg_dict.items():
            if len(segs) == 0:
                continue
            chosen = segs[np.random.choice(len(segs))]
            feats = extract_wpd_features(chosen)
            X.append(feats)
            y.append(0 if condition == "EC" else 1)
            subjects.append(subject_id)

    return np.asarray(X), np.asarray(y), np.asarray(subjects)


# =========================
# SUBJECT-WISE CV EVALUATION
# =========================
def subjectwise_cv_all_models(X, y, subject_ids):
    unique_subjects = np.unique(subject_ids)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {"DNN": [], "SVM": [], "RF": [], "GBM": []}

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

        # Feature selection (fit only train)
        selector = SelectPercentile(score_func=f_classif, percentile=FEATURE_PERCENTILE)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # ---------- SVM ----------
        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        results["SVM"].append(accuracy_score(y_test, pred))

        # ---------- RF ----------
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        results["RF"].append(accuracy_score(y_test, pred))

        # ---------- GBM ----------
        gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
        gbm.fit(X_train, y_train)
        pred = gbm.predict(X_test)
        results["GBM"].append(accuracy_score(y_test, pred))

        # ---------- DNN ----------
        dnn = build_dnn(X_train.shape[1])
        es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

        dnn.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.1,
            callbacks=[es],
            verbose=0
        )
        pred = (dnn.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
        results["DNN"].append(accuracy_score(y_test, pred))

        print(f"\nFold {fold}/{KFOLDS}")
        print(f"  DNN: {results['DNN'][-1]:.3f}")
        print(f"  SVM: {results['SVM'][-1]:.3f}")
        print(f"  RF : {results['RF'][-1]:.3f}")
        print(f"  GBM: {results['GBM'][-1]:.3f}")

    print("\n====================")
    print("Mean Accuracy (CV)")
    print("====================")
    for k, v in results.items():
        print(f"{k:>4}: {np.mean(v):.3f}  (+/- {np.std(v):.3f})")

    return results


def main():
    X, y, subject_ids = build_subjectwise_dataset(DATA_DIR)

    print("X shape:", X.shape)
    print("y distribution (EC=0, EO=1):", np.bincount(y))

    if len(np.unique(y)) < 2:
        raise ValueError("Only one class found. Check filenames contain EC/EO correctly.")

    subjectwise_cv_all_models(X, y, subject_ids)


if __name__ == "__main__":
    main()

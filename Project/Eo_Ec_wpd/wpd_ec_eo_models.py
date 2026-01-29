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
# Config
# =========================
SAMPLING_RATE = 250
SEGMENT_SEC = 30
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SEC

DATA_DIR = r"C:\research\EEG_Domain\combined\combined_new"  # change if needed

WAVELET = "db4"
WPD_LEVEL = 5

FS_PERCENT = 60  # ANOVA feature selection percentile
KFOLDS = 10
RANDOM_STATE = 42


# =========================
# Data loading + segmentation
# =========================
def load_eeg_segments(csv_path: str):
    """
    Read EEG CSV, convert to numeric, segment into fixed windows.
    Label is inferred from filename: contains 'EC' => EC else EO.
    """
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    label_str = "EC" if "EC" in os.path.basename(csv_path) else "EO"

    segments = []
    for start in range(0, len(df), SEGMENT_LEN):
        seg = df.iloc[start:start + SEGMENT_LEN]
        if len(seg) == SEGMENT_LEN:
            segments.append(seg)

    return segments, label_str


# =========================
# WPD feature extraction
# =========================
def extract_wpd_features(segment: pd.DataFrame):
    """
    Wavelet Packet Decomposition (WPD) feature extraction.

    For each channel:
      - WPD decomposition (db4, maxlevel=5)
      - For each node at level 5 (natural order):
          mean, var, RMS, kurtosis, skew
    """
    feats = []

    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)

        packet = pywt.WaveletPacket(
            data=x,
            wavelet=WAVELET,
            mode="symmetric",
            maxlevel=WPD_LEVEL
        )

        nodes = packet.get_level(WPD_LEVEL, order="natural")
        for node in nodes:
            node_data = np.asarray(node.data, dtype=float)

            feats.extend([
                float(np.mean(node_data)),
                float(np.var(node_data)),
                float(np.sqrt(np.mean(node_data ** 2))),  # RMS
                float(kurtosis(node_data)),
                float(skew(node_data))
            ])

    return np.asarray(feats, dtype=float)


# =========================
# Models
# =========================
def build_dnn(input_dim: int):
    model = Sequential([
        Dense(100, activation="relu", input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(50, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_subjectwise_cv(X, y, subject_ids):
    """
    Subject-wise KFold:
      - split on unique subject IDs
      - train/test masks are formed from subject membership
    """
    unique_subjects = np.unique(subject_ids)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "SVM": [],
        "RF": [],
        "GBM": [],
        "DNN": []
    }

    for fold, (train_sub_idx, test_sub_idx) in enumerate(kf.split(unique_subjects), start=1):
        train_subjects = unique_subjects[train_sub_idx]
        test_subjects = unique_subjects[test_sub_idx]

        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Scale (fit only on train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature selection (fit only on train)
        selector = SelectPercentile(score_func=f_classif, percentile=FS_PERCENT)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # -----------------
        # SVM
        # -----------------
        svm = SVC(kernel="linear", random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        results["SVM"].append(accuracy_score(y_test, y_pred))

        # -----------------
        # Random Forest
        # -----------------
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results["RF"].append(accuracy_score(y_test, y_pred))

        # -----------------
        # Gradient Boosting
        # -----------------
        gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
        gbm.fit(X_train, y_train)
        y_pred = gbm.predict(X_test)
        results["GBM"].append(accuracy_score(y_test, y_pred))

        # -----------------
        # DNN
        # -----------------
        dnn = build_dnn(X_train.shape[1])
        es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

        dnn.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[es],
            verbose=0
        )
        y_prob = dnn.predict(X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int).ravel()
        results["DNN"].append(accuracy_score(y_test, y_pred))

        print(f"\nFold {fold}/{KFOLDS}")
        print(f"  SVM Acc: {results['SVM'][-1]:.4f}")
        print(f"  RF  Acc: {results['RF'][-1]:.4f}")
        print(f"  GBM Acc: {results['GBM'][-1]:.4f}")
        print(f"  DNN Acc: {results['DNN'][-1]:.4f}")

    return results


def main():
    all_features = []
    all_labels = []
    subject_ids = []

    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(DATA_DIR, fn)
        subject_id = fn.split("_")[0]

        segments, label_str = load_eeg_segments(csv_path)

        for seg in segments:
            feats = extract_wpd_features(seg)
            all_features.append(feats)
            all_labels.append(0 if label_str == "EC" else 1)
            subject_ids.append(subject_id)

    X = np.asarray(all_features, dtype=float)
    y = np.asarray(all_labels, dtype=int)
    subject_ids = np.asarray(subject_ids)

    print("Feature matrix:", X.shape)
    print("Label counts (EC=0, EO=1):", np.bincount(y))

    results = run_subjectwise_cv(X, y, subject_ids)

    print("\n====================")
    print("Mean CV Accuracy")
    print("====================")
    for model_name, accs in results.items():
        print(f"{model_name:>4}: {np.mean(accs):.4f}  (+/- {np.std(accs):.4f})")


if __name__ == "__main__":
    main()

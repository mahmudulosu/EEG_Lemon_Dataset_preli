import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping


SAMPLING_RATE = 250
SEGMENT_SEC = 30
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SEC
DATA_DIR = r"C:\research\EEG_Domain\combined\combined_new"
FS_PERCENT = 60
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


def extract_dwt_features(segment: pd.DataFrame, wavelet="db4", level=6):
    feats = []
    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)
        coeffs = pywt.wavedec(x, wavelet, level=level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            feats.extend([
                float(np.std(c)),
                float(np.mean(c)),
                float(np.sqrt(np.mean(c**2))),
                float(np.var(c)),
                float(kurtosis(c)),
                float(skew(c)),
            ])
    return np.asarray(feats, dtype=float)


def build_cnn1d(input_len: int):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(input_len, 1)),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    X, y, subjects = [], [], []

    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".csv"):
            continue

        label_str = "EC" if "EC" in fn else "EO"
        label = 0 if label_str == "EC" else 1
        subject_id = fn.split("_")[0]

        for seg in load_segments(os.path.join(DATA_DIR, fn)):
            X.append(extract_dwt_features(seg))
            y.append(label)
            subjects.append(subject_id)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects)

    # Subject-wise split
    unique_subjects = np.unique(subjects)
    train_subj, test_subj = train_test_split(unique_subjects, test_size=0.2, random_state=RANDOM_STATE)

    train_mask = np.isin(subjects, train_subj)
    test_mask = np.isin(subjects, test_subj)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Scale + feature selection
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = SelectPercentile(f_classif, percentile=FS_PERCENT)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # CNN expects shape: (N, length, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = build_cnn1d(input_len=X_train.shape[1])
    es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["EC", "EO"], zero_division=0))


if __name__ == "__main__":
    main()

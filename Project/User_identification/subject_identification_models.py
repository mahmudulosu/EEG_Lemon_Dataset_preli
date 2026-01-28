import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# Config
# -----------------------------
SAMPLING_RATE = 250
SEGMENT_SEC = 2
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SEC

DATA_FOLDER = r"C:\research\common_data_eeg_eo"  # <-- change this to your folder

WAVELET = "db4"
DWT_LEVEL = 5
FS_PERCENTILE = 60
RANDOM_STATE = 42


# -----------------------------
# Data Loading / Feature Extraction
# -----------------------------
def safe_wavedec(x: np.ndarray, wavelet: str, level: int):
    """Clip wavelet level to avoid 'level too high' errors."""
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    use_level = min(level, max_level)
    return pywt.wavedec(x, wavelet, level=use_level)

def load_eeg_segments(csv_path: str, segment_len: int):
    """Load one CSV and split into non-overlapping segments."""
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    segments = []
    for start in range(0, len(df), segment_len):
        end = start + segment_len
        seg = df.iloc[start:end]
        if len(seg) == segment_len:
            segments.append(seg)
    return segments

def extract_dwt_features(segment: pd.DataFrame, wavelet=WAVELET, level=DWT_LEVEL):
    """DWT features per channel: mean, var, RMS for each coefficient array."""
    feats = []
    for col in segment.columns:
        x = segment[col].to_numpy(dtype=float)
        coeffs = safe_wavedec(x, wavelet, level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            feats.extend([
                float(np.mean(c)),
                float(np.var(c)),
                float(np.sqrt(np.mean(c ** 2))),  # RMS
            ])
    return np.asarray(feats, dtype=float)

def build_dataset(data_folder: str):
    """
    Builds segment-level dataset and does per-subject 80/20 split like your code:
    - each subject contributes segments
    - split segments within subject into train/test
    """
    all_train_X, all_train_y = [], []
    all_test_X, all_test_y = [], []

    # First pass: collect data by subject
    for fn in os.listdir(data_folder):
        if not fn.lower().endswith(".csv"):
            continue

        subject_id = fn.split("_")[0]
        fp = os.path.join(data_folder, fn)

        segments = load_eeg_segments(fp, SEGMENT_LEN)
        if len(segments) < 2:
            continue

        X = [extract_dwt_features(seg) for seg in segments]
        y = [subject_id] * len(X)

        # Per-subject split (within-session)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        all_train_X.extend(X_train)
        all_train_y.extend(y_train)
        all_test_X.extend(X_test)
        all_test_y.extend(y_test)

    all_train_X = np.asarray(all_train_X, dtype=float)
    all_test_X = np.asarray(all_test_X, dtype=float)
    all_train_y = np.asarray(all_train_y)
    all_test_y = np.asarray(all_test_y)

    # Encode subject IDs
    unique_subjects = np.array(sorted(np.unique(all_train_y)))
    label_map = {sid: i for i, sid in enumerate(unique_subjects)}

    y_train_enc = np.array([label_map[s] for s in all_train_y], dtype=int)
    y_test_enc = np.array([label_map[s] for s in all_test_y], dtype=int)

    return all_train_X, y_train_enc, all_test_X, y_test_enc, unique_subjects


# -----------------------------
# Shared preprocessing (fit on train only)
# -----------------------------
def preprocess_train_test(X_train, y_train, X_test, percentile=FS_PERCENTILE):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)

    return X_train_sel, X_test_sel


# -----------------------------
# Models
# -----------------------------
def train_predict_svm(X_train, y_train, X_test):
    model = SVC(kernel="linear", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_predict_rf(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_predict_gbm(X_train, y_train, X_test):
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

def build_dnn(input_dim: int, num_classes: int):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_predict_dnn(X_train, y_train, X_test, num_classes: int):
    model = build_dnn(X_train.shape[1], num_classes=num_classes)
    es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )
    probs = model.predict(X_test, verbose=0)
    return np.argmax(probs, axis=1)


# -----------------------------
# Evaluation + Plot
# -----------------------------
def evaluate_model(name, y_true, y_pred, target_names, plot_cm=True):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    if plot_cm:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


def main():
    X_train, y_train, X_test, y_test, target_names = build_dataset(DATA_FOLDER)
    print("[DATA] Train:", X_train.shape, " Test:", X_test.shape)
    print("[DATA] #Subjects:", len(target_names))

    # Preprocess (scale + ANOVA selection)
    X_train_sel, X_test_sel = preprocess_train_test(X_train, y_train, X_test, percentile=FS_PERCENTILE)

    # SVM
    y_pred_svm = train_predict_svm(X_train_sel, y_train, X_test_sel)
    evaluate_model("Linear SVM", y_test, y_pred_svm, target_names)

    # Random Forest
    y_pred_rf = train_predict_rf(X_train_sel, y_train, X_test_sel)
    evaluate_model("Random Forest", y_test, y_pred_rf, target_names, plot_cm=False)

    # Gradient Boosting
    y_pred_gbm = train_predict_gbm(X_train_sel, y_train, X_test_sel)
    evaluate_model("Gradient Boosting", y_test, y_pred_gbm, target_names, plot_cm=False)

    # DNN
    y_pred_dnn = train_predict_dnn(X_train_sel, y_train, X_test_sel, num_classes=len(target_names))
    evaluate_model("DNN (Keras)", y_test, y_pred_dnn, target_names, plot_cm=False)


if __name__ == "__main__":
    main()

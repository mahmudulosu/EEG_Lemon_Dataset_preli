import os
import numpy as np
import pandas as pd
import pywt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# --------------------
# Config
# --------------------
SAMPLING_RATE = 250
SEGMENT_SEC = 5
SEGMENT_LEN = SAMPLING_RATE * SEGMENT_SEC

DATA_FOLDER = r"C:\research\common_data_eeg_ec"   # EC or EO folder
META_CSV = r"C:\research\MRI\participants_LSD_andLEMON.csv"

WAVELET = "db4"
LEVEL = 5
FS_PERCENT = 60
N_SPLITS = 10
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
            c = np.asarray(c, dtype=float)
            feats.extend([float(np.mean(c)), float(np.var(c)), float(np.sqrt(np.mean(c**2)))])
    return np.asarray(feats, dtype=float)


def build_dnn(input_dim: int):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def run_subject_cv(model_name, model_builder, X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_acc = []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scale + feature selection (fit only on train fold)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        selector = SelectPercentile(f_classif, percentile=FS_PERCENT)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        if model_name == "DNN":
            dnn = model_builder(X_train.shape[1])
            es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
            dnn.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
            y_pred = (dnn.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
        else:
            clf = model_builder()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        fold_acc.append(accuracy_score(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    print(f"\n===== {model_name} (Subject-wise CV) =====")
    print("Fold Accuracies:", [f"{a*100:.2f}%" for a in fold_acc])
    print(f"Mean Accuracy: {np.mean(fold_acc)*100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true_all, y_pred_all))
    print("\nClassification Report:\n",
          classification_report(y_true_all, y_pred_all, target_names=["Female", "Male"], zero_division=0))


def main():
    meta = pd.read_csv(META_CSV)
    gender_map = {str(r["participant_id"]): str(r["gender"]).upper() for _, r in meta.iterrows()}

    X, y, groups = [], [], []
    for fn in os.listdir(DATA_FOLDER):
        if not fn.lower().endswith(".csv"):
            continue

        subject_id = fn.split("_")[0]
        if subject_id not in gender_map:
            continue

        gender = gender_map[subject_id]
        label = 0 if gender == "F" else 1

        for seg in load_segments(os.path.join(DATA_FOLDER, fn)):
            X.append(extract_dwt_features(seg))
            y.append(label)
            groups.append(subject_id)  # group by subject

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    # Models
    run_subject_cv("SVM", lambda: SVC(kernel="linear", random_state=RANDOM_STATE), X, y, groups)
    run_subject_cv("RandomForest", lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), X, y, groups)
    run_subject_cv("GBM", lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE), X, y, groups)
    run_subject_cv("DNN", lambda input_dim: build_dnn(input_dim), X, y, groups)


if __name__ == "__main__":
    main()

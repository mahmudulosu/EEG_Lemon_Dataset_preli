import os
import argparse
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# Feature Extraction
# =========================
def load_eeg_data(file_path: str):
    """Load one EEG CSV and infer label from filename."""
    df = pd.read_csv(file_path)
    label = "EC" if "EC" in os.path.basename(file_path) else "EO"
    return df, label

def extract_dwt_features(df: pd.DataFrame, wavelet="db4", level=6):
    """
    DWT features per channel.
    For each DWT coefficient array compute:
      std, mean, RMS, var, kurtosis, skew
    """
    features = []
    for col in df.columns:
        signal = df[col].to_numpy(dtype=float)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        for c in coeffs:
            c = np.asarray(c, dtype=float)
            features.extend([
                np.std(c),
                np.mean(c),
                np.sqrt(np.mean(c ** 2)),
                np.var(c),
                kurtosis(c),
                skew(c)
            ])
    return np.asarray(features, dtype=float)

def build_dataset(data_dir: str, wavelet="db4", level=6):
    """
    Builds dataset where each CSV file contributes one feature vector.
    subject_id is assumed to be the filename prefix before first underscore.
    Example: 'S01_EC.csv' -> subject_id = 'S01'
    """
    X_list, y_list, sid_list = [], [], []

    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue

        fp = os.path.join(data_dir, fn)
        subject_id = fn.split("_")[0]

        df, label_str = load_eeg_data(fp)
        feats = extract_dwt_features(df, wavelet=wavelet, level=level)

        X_list.append(feats)
        y_list.append(label_str)
        sid_list.append(subject_id)

    X = np.asarray(X_list, dtype=float)
    y_str = np.asarray(y_list)
    subject_ids = np.asarray(sid_list)

    label_map = {"EC": 0, "EO": 1}
    y = np.array([label_map[v] for v in y_str], dtype=int)

    return X, y, subject_ids


# =========================
# Models
# =========================
def build_dnn(input_dim: int, dropout=0.5):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation="relu"),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def fit_predict_dnn(X_train, y_train, X_test, epochs=150, batch_size=32, dropout=0.5, patience=10, val_split=0.1):
    """
    DNN training with early stopping.
    Assumes X_train/X_test are already scaled and selected.
    """
    model = build_dnn(X_train.shape[1], dropout=dropout)
    es = EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[es],
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred


def make_sklearn_pipeline(model_name: str, feature_percentile=60, random_state=42):
    """
    Returns a sklearn Pipeline:
      feature selection (ANOVA) -> scaling -> model
    """
    selector = SelectPercentile(score_func=f_classif, percentile=feature_percentile)

    if model_name == "svm":
        model = SVC(kernel="linear", C=1, random_state=random_state)
        scaler = StandardScaler()

    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        scaler = StandardScaler()

    elif model_name == "gbm":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
        scaler = StandardScaler()

    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, activation="relu",
                              solver="adam", random_state=random_state)
        scaler = StandardScaler()

    elif model_name == "rbm":
        # RBM works better with values in [0,1]
        rbm1 = BernoulliRBM(n_components=256, learning_rate=0.05, n_iter=10, random_state=random_state)
        rbm2 = BernoulliRBM(n_components=128, learning_rate=0.05, n_iter=10, random_state=random_state)
        logistic = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        model = Pipeline([("rbm1", rbm1), ("rbm2", rbm2), ("logistic", logistic)])
        scaler = MinMaxScaler()

    elif model_name == "pydbn":
        # Optional dependency: pydbn
        try:
            from pydbn.models import SupervisedDBNClassification
        except Exception as e:
            raise ImportError(
                "pydbn is not installed or not working. Install it or choose another model.\n"
                "Try: pip install pydbn"
            ) from e

        model = SupervisedDBNClassification(
            hidden_layers_structure=[256, 128],
            learning_rate_rbm=0.05,
            learning_rate=0.1,
            n_epochs_rbm=10,
            n_iter_backprop=100,
            batch_size=32,
            activation_function="relu",
            dropout_p=0.2
        )
        scaler = StandardScaler()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("select", selector),
        ("scale", scaler),
        ("model", model),
    ])


# =========================
# Evaluation
# =========================
def subject_holdout_eval(X, y, subject_ids, model_name, feature_percentile=60, seed=42):
    """
    Subject-wise train/test split:
      split subjects -> build masks -> train on train subjects, test on test subjects.
    """
    unique_subjects = np.unique(subject_ids)
    train_subj, test_subj = train_test_split(unique_subjects, test_size=0.2, random_state=seed)

    train_mask = np.isin(subject_ids, train_subj)
    test_mask = np.isin(subject_ids, test_subj)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("[Holdout] y_test distribution (EC=0, EO=1):", np.bincount(y_test) if len(y_test) else "empty")

    if model_name == "dnn":
        # IMPORTANT: fit selector+scaler on train only (no leakage)
        selector = SelectPercentile(score_func=f_classif, percentile=feature_percentile)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        y_pred = fit_predict_dnn(X_train_s, y_train, X_test_s)
    else:
        pipe = make_sklearn_pipeline(model_name, feature_percentile=feature_percentile, random_state=seed)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy ({model_name.upper()}): {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["EC", "EO"]))
    return acc


def subject_kfold_eval(X, y, subject_ids, model_name, k=10, feature_percentile=60, seed=42):
    """
    Subject-wise K-fold cross-validation using GroupKFold.
    Each subject appears in ONLY one fold (no leakage).
    """
    gkf = GroupKFold(n_splits=k)
    accs = []

    fold = 0
    for tr_idx, te_idx in gkf.split(X, y, groups=subject_ids):
        fold += 1
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]

        # Some folds can become single-class if dataset is small
        if len(np.unique(y_test)) < 2:
            print(f"Fold {fold:02d}/{k} skipped (only one class in test fold).")
            continue

        if model_name == "dnn":
            selector = SelectPercentile(score_func=f_classif, percentile=feature_percentile)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_test_s = scaler.transform(X_test_sel)

            y_pred = fit_predict_dnn(X_train_s, y_train, X_test_s)
        else:
            pipe = make_sklearn_pipeline(model_name, feature_percentile=feature_percentile, random_state=seed)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        print(f"Fold {fold:02d} accuracy: {acc:.4f} | test dist: {np.bincount(y_test)}")

    if len(accs) == 0:
        print("No valid folds were evaluated.")
        return None

    print(f"\nMean CV Accuracy ({model_name.upper()}): {np.mean(accs):.4f} (n={len(accs)} folds)")
    return float(np.mean(accs))


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing EEG CSV files.")
    parser.add_argument("--model", type=str, default="dnn",
                        choices=["dnn", "svm", "rf", "gbm", "mlp", "rbm", "pydbn"],
                        help="Model to train.")
    parser.add_argument("--eval", type=str, default="holdout",
                        choices=["holdout", "kfold"],
                        help="Evaluation method: subject-wise holdout or subject-wise k-fold CV.")
    parser.add_argument("--k", type=int, default=10, help="Number of folds (only for kfold).")
    parser.add_argument("--percentile", type=int, default=60, help="ANOVA feature selection percentile.")
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet family for DWT.")
    parser.add_argument("--level", type=int, default=6, help="DWT decomposition level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    X, y, subject_ids = build_dataset(args.data_dir, wavelet=args.wavelet, level=args.level)
    print("[Data] X shape:", X.shape)
    print("[Data] Label distribution (EC=0, EO=1):", np.bincount(y))
    print("[Data] Unique subjects:", len(np.unique(subject_ids)))

    if args.eval == "holdout":
        subject_holdout_eval(X, y, subject_ids, args.model, feature_percentile=args.percentile, seed=args.seed)
    else:
        subject_kfold_eval(X, y, subject_ids, args.model, k=args.k,
                           feature_percentile=args.percentile, seed=args.seed)


if __name__ == "__main__":
    main()

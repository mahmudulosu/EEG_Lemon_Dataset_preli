from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from .models import make_early_stopping

def segment_level_stratified_cv_dnn(
    X: np.ndarray,
    y: np.ndarray,
    build_model_fn,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    epochs: int = 150,
    batch_size: int = 32,
    val_split: float = 0.1,
    patience: int = 10,
):
    """
    Segment-level CV (StratifiedKFold on samples). Can leak subjects if multiple segments per subject exist.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_accs = []

    for fold, (tr, te) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_model_fn(input_dim=X_train_s.shape[1])
        es = make_early_stopping(patience=patience)

        model.fit(
            X_train_s, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[es],
            verbose=0
        )

        y_prob = model.predict(X_test_s, verbose=0)
        if y_prob.shape[1:] == ():  # binary -> (n,1) sometimes
            y_pred = (y_prob.reshape(-1) > 0.5).astype(int)
        elif y_prob.shape[1] == 1:
            y_pred = (y_prob.reshape(-1) > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        print(f"Fold {fold:02d} accuracy: {acc:.4f}")

    return np.array(fold_accs)

def subject_holdout_split_dnn(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    build_model_fn,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 150,
    batch_size: int = 32,
    val_split: float = 0.1,
    patience: int = 10,
):
    """
    Subject-wise holdout split: no subject leakage.
    """
    unique_subjects = np.unique(subject_ids)
    train_subj, test_subj = train_test_split(unique_subjects, test_size=test_size, random_state=random_state)

    train_mask = np.isin(subject_ids, train_subj)
    test_mask = np.isin(subject_ids, test_subj)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_model_fn(input_dim=X_train_s.shape[1])
    es = make_early_stopping(patience=patience)

    model.fit(
        X_train_s, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[es],
        verbose=0
    )

    y_prob = model.predict(X_test_s, verbose=0)
    if y_prob.shape[1] == 1:
        y_pred = (y_prob.reshape(-1) > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    return acc

def subject_groupkfold_cv_sklearn_model(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_ctor,
    n_splits: int = 10,
):
    """
    Subject-wise CV for sklearn models using GroupKFold.
    Ensures subjects do not appear in both train and test.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), start=1):
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_ctor()
        model.fit(X_train_s, y_train)
        acc = model.score(X_test_s, y_test)

        fold_accs.append(acc)
        print(f"Fold {fold:02d} accuracy: {acc:.4f}")

    return np.array(fold_accs)

def save_fold_results_csv(fold_accs: np.ndarray, out_path: str, label: str):
    df = pd.DataFrame({
        "fold": np.arange(1, len(fold_accs) + 1),
        "accuracy": fold_accs
    })
    df.loc[len(df)] = ["mean", float(np.mean(fold_accs))]
    df.to_csv(out_path, index=False)
    print(f"[Saved] {label} -> {out_path}")

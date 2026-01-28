import argparse
import yaml
import os

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from eeg_dwt.utils import set_seed, ensure_dir
from eeg_dwt.data import (
    load_binary_ec_eo_segments,
    build_pid_to_label_4class,
    load_multiclass_segments,
)
from eeg_dwt.features import build_feature_matrix
from eeg_dwt.selection import select_features_anova
from eeg_dwt.evaluation import subject_groupkfold_cv_sklearn_model, save_fold_results_csv

def main(cfg, task: str):
    set_seed(cfg["cv"]["random_state"])
    segment_length = int(cfg["sampling_rate"] * cfg["segment_seconds"])

    if task == "binary":
        segments, y, subject_ids = load_binary_ec_eo_segments(cfg["eeg_dir"], segment_length)
    else:
        pid_to_label = build_pid_to_label_4class(cfg["meta_csv"], cfg["age_bins"])
        segments, y, subject_ids = load_multiclass_segments(cfg["eeg_dir"], segment_length, pid_to_label)

    X = build_feature_matrix(segments, wavelet=cfg["dwt_wavelet"], level=cfg["dwt_level"])
    X_sel, _ = select_features_anova(X, y, percentile=cfg["feature_select_percentile"])

    ensure_dir(cfg["output_dir"])

    # SVM
    svm_ctor = lambda: SVC(kernel="linear", C=1, random_state=cfg["cv"]["random_state"])
    svm_accs = subject_groupkfold_cv_sklearn_model(X_sel, y, subject_ids, svm_ctor, n_splits=cfg["cv"]["n_splits"])
    save_fold_results_csv(svm_accs, os.path.join(cfg["output_dir"], f"{task}_svm_subject_groupkfold.csv"), "SVM subject CV")

    # Random Forest
    rf_ctor = lambda: RandomForestClassifier(n_estimators=200, random_state=cfg["cv"]["random_state"])
    rf_accs = subject_groupkfold_cv_sklearn_model(X_sel, y, subject_ids, rf_ctor, n_splits=cfg["cv"]["n_splits"])
    save_fold_results_csv(rf_accs, os.path.join(cfg["output_dir"], f"{task}_rf_subject_groupkfold.csv"), "RF subject CV")

    # Gradient Boosting
    gbm_ctor = lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=cfg["cv"]["random_state"])
    gbm_accs = subject_groupkfold_cv_sklearn_model(X_sel, y, subject_ids, gbm_ctor, n_splits=cfg["cv"]["n_splits"])
    save_fold_results_csv(gbm_accs, os.path.join(cfg["output_dir"], f"{task}_gbm_subject_groupkfold.csv"), "GBM subject CV")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.task)

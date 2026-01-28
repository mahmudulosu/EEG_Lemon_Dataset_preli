import argparse
import yaml
import os
import numpy as np

from eeg_dwt.utils import set_seed, ensure_dir
from eeg_dwt.data import load_binary_ec_eo_segments
from eeg_dwt.features import build_feature_matrix
from eeg_dwt.selection import select_features_anova
from eeg_dwt.models import build_dnn_binary
from eeg_dwt.evaluation import (
    segment_level_stratified_cv_dnn,
    subject_holdout_split_dnn,
    save_fold_results_csv,
)

def main(cfg):
    set_seed(cfg["cv"]["random_state"])

    segment_length = int(cfg["sampling_rate"] * cfg["segment_seconds"])

    segments, y, subject_ids = load_binary_ec_eo_segments(
        eeg_dir=cfg["eeg_dir"],
        segment_length=segment_length
    )
    print("[Data] segments:", len(segments), "labels:", np.bincount(y))

    X = build_feature_matrix(segments, wavelet=cfg["dwt_wavelet"], level=cfg["dwt_level"])
    print("[Features] X shape:", X.shape)

    X_sel, _ = select_features_anova(X, y, percentile=cfg["feature_select_percentile"])
    print("[FS] X_sel shape:", X_sel.shape)

    def model_fn(input_dim):
        return build_dnn_binary(input_dim=input_dim, dropout=cfg["dnn"]["dropout"])

    # Segment-level CV
    fold_accs = segment_level_stratified_cv_dnn(
        X_sel, y, model_fn,
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
        epochs=cfg["dnn"]["epochs"],
        batch_size=cfg["dnn"]["batch_size"],
        val_split=cfg["dnn"]["val_split"],
        patience=cfg["dnn"]["early_stopping_patience"],
    )

    ensure_dir(cfg["output_dir"])
    out_csv = os.path.join(cfg["output_dir"], "binary_ec_eo_dnn_segment_cv.csv")
    save_fold_results_csv(fold_accs, out_csv, label="Binary DNN segment CV")

    # Subject-wise holdout
    subj_acc = subject_holdout_split_dnn(
        X_sel, y, subject_ids, model_fn,
        test_size=0.2,
        random_state=cfg["cv"]["random_state"],
        epochs=cfg["dnn"]["epochs"],
        batch_size=cfg["dnn"]["batch_size"],
        val_split=cfg["dnn"]["val_split"],
        patience=cfg["dnn"]["early_stopping_patience"],
    )
    print(f"[Subject Holdout] accuracy: {subj_acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)

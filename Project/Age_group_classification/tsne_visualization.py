import os
import argparse
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.manifold import TSNE


def build_age_mapping(csv_path: str):
    meta = pd.read_csv(csv_path)
    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {
        "20-25": "young",
        "25-30": "young",
        "60-65": "old",
        "65-70": "old",
        "70-75": "old",
    }
    meta["age"] = meta["age"].astype(str).str.strip()
    meta["participant_id"] = meta["participant_id"].astype(str).str.strip()
    meta = meta[meta["age"].isin(valid_ages)]
    return {row["participant_id"]: age_group_mapping[row["age"]] for _, row in meta.iterrows()}


def safe_wavedec(x: np.ndarray, wavelet: str, level: int):
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=w.dec_len)
    use_level = min(level, max_level)
    return pywt.wavedec(x, wavelet, level=use_level)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def dwt_features(df: pd.DataFrame, wavelet="db4", level=5):
    feats = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        coeffs = safe_wavedec(x, wavelet, level)
        for c in coeffs:
            feats.extend([np.mean(c), np.var(c), np.sqrt(np.mean(np.asarray(c) ** 2))])
    return np.asarray(feats, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="t-SNE visualization (young vs old) from EO+EC subject-level features.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--percentile", type=int, default=60)
    ap.add_argument("--max_subjects", type=int, default=60, help="Plot up to N subjects for readability.")
    args = ap.parse_args()

    age_mapping = build_age_mapping(args.meta_csv)

    # load EO+EC per subject and concat features
    by_subj = {}
    for fn in os.listdir(args.data_dir):
        if not fn.endswith(".csv"):
            continue
        sid = fn.split("_")[0]
        cond = fn.split("_")[1].split(".")[0]
        if sid not in age_mapping:
            continue
        by_subj.setdefault(sid, {})
        by_subj[sid][cond] = load_csv(os.path.join(args.data_dir, fn))

    X, y = [], []
    for sid, conds in by_subj.items():
        if "EO" not in conds or "EC" not in conds:
            continue
        eo, ec = conds["EO"], conds["EC"]
        common = [c for c in eo.columns if c in ec.columns]
        if not common:
            continue
        eo, ec = eo[common], ec[common]
        n = min(len(eo), len(ec))
        eo, ec = eo.iloc[:n], ec.iloc[:n]

        feats = np.concatenate([
            dwt_features(eo, args.wavelet, args.level),
            dwt_features(ec, args.wavelet, args.level),
        ])

        X.append(feats)
        y.append(0 if age_mapping[sid] == "young" else 1)

        if len(X) >= args.max_subjects:
            break

    X = np.asarray(X, float)
    y = np.asarray(y, int)

    # scale + select
    Xs = StandardScaler().fit_transform(X)
    Xs = SelectPercentile(f_classif, percentile=args.percentile).fit_transform(Xs, y)

    emb = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(Xs)

    plt.figure(figsize=(9, 6))
    for cls, name in [(0, "young"), (1, "old")]:
        idx = (y == cls)
        plt.scatter(emb[idx, 0], emb[idx, 1], label=name, s=40)

    plt.title("t-SNE (subject-level features): young vs old")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

from __future__ import annotations
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif

def select_features_anova(X: np.ndarray, y: np.ndarray, percentile: int = 60):
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    X_sel = selector.fit_transform(X, y)
    return X_sel, selector

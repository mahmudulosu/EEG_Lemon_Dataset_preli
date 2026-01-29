import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE


# =========================
# CONFIG
# =========================
CSV_PATH = r"C:\Users\mdmahha\Downloads\eye_data.csv"  # <-- change to your path
N_FEATURES = 14          # first 14 columns are features
LABEL_COL = 14           # column index of label (15th column)
RANDOM_STATE = 42

# Grid search space (you can expand this if you want)
PARAM_GRID = {
    "C": [1, 10, 100, 1000],
    "gamma": [1e3, 1e4, 1e5, 1e6],
    "kernel": ["rbf"]
}

# Dataset split ratios
TEST_SIZE = 0.23         # roughly matches your 10k train / 3k test if data ~13k
VAL_SIZE_WITHIN_TEST = 0.33  # split the test chunk into val/test


# =========================
# HELPERS
# =========================
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X = df.iloc[:, 0:N_FEATURES].to_numpy(dtype=float)
    y = df.iloc[:, LABEL_COL].to_numpy().ravel()

    # Ensure labels are numeric (some versions store as strings)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le


def plot_tsne(X, y, title="t-SNE plot (colored by label)"):
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():
    X, y, label_encoder = load_data(CSV_PATH)

    # Train/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Validation/Test split (from the temp set)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - VAL_SIZE_WITHIN_TEST),
        random_state=RANDOM_STATE, stratify=y_temp
    )

    # Normalize feature vectors (L2 normalization)
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    print("Shapes:")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Label distribution (train):", np.bincount(y_train))

    # GridSearchCV for SVM
    base_svm = SVC()
    grid = GridSearchCV(
        estimator=base_svm,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=2,
        verbose=2,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    best_model = grid.best_estimator_

    # Validation evaluation
    y_val_pred = best_model.predict(X_val)
    print("\n=== Validation Results ===")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    # Test evaluation
    y_test_pred = best_model.predict(X_test)
    print("\n=== Test Results ===")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))

    # t-SNE visualization (use a subset if dataset is large to speed up)
    # Here we visualize training data
    print("\nGenerating t-SNE plot...")
    plot_tsne(X_train, y_train, title="t-SNE: UCI EEG Eye State (Train set)")


if __name__ == "__main__":
    main()

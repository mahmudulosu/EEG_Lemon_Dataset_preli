from __future__ import annotations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_dnn_binary(input_dim: int, dropout: float = 0.5):
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

def build_dnn_multiclass(input_dim: int, num_classes: int = 4, dropout: float = 0.5):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation="relu"),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def make_early_stopping(patience: int = 10):
    return EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)

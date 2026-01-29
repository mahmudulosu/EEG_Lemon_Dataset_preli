EEG EC vs EO Classification using Wavelet Packet Decomposition (WPD)

This project performs binary classification of resting-state EEG into:

EC = Eyes Closed

EO = Eyes Open

It uses Wavelet Packet Decomposition (WPD) to extract time–frequency features from EEG windows, then trains multiple classifiers using subject-wise cross-validation.

Why WPD?

Unlike standard DWT (Discrete Wavelet Transform), Wavelet Packet Decomposition decomposes both approximation and detail components at each level, providing a richer frequency partition. This often helps EEG tasks where discriminative information can be spread across multiple frequency sub-bands.

Pipeline

Load EEG CSV files (multi-channel EEG time series)

Segment each recording into fixed windows (default: 30 seconds @ 250 Hz)

WPD Feature Extraction

Wavelet: db4

Level: 5

For each channel and each node at level 5, compute:

mean, variance, RMS, kurtosis, skewness

Feature Selection

ANOVA F-test (SelectPercentile, keep top 60%)

Scaling

Standardization (StandardScaler)

Subject-wise 10-fold Cross-Validation

Subjects in test folds do not appear in training folds

Models

DNN (Keras)

SVM (linear)

Random Forest

Gradient Boosting

Input data format

EEG files are expected as .csv inside data_dir.

File naming must include either EC or EO so labels can be inferred.

Subject ID is assumed to be the part before the first underscore:

Example: S001_EC.csv → subject=S001, label=EC

# -*- coding: utf-8 -*-
"""
Demo script for DDFGB-BLS.
Usage: Place binary classification .mat dataset(s) into './datasets/' and run this script.
"""

import os
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

from ddfgb_bls import compute_unified_fuzzy_scores, ddfgb_bls_predict, ddfgb_bls_train
from granular_ball import get_gb_details
from metrics import compute_metrics


def load_mat_dataset(file_path):
    mat_data = sio.loadmat(file_path)
    key = [k for k in mat_data.keys() if not k.startswith('_')][0]
    data = mat_data[key]
    X, labels = data[:, :-1], data[:, -1]

    # Map minority class to +1 and majority class to -1
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError(f"Expected binary classification dataset, got {len(unique_classes)} classes.")
    minority_class = unique_classes[np.argmin(class_counts)]
    y = np.where(labels == minority_class, 1, -1)
    return X, y


def run_single_dataset(file_path, noise_rate=0.20):
    np.random.seed(0)
    random.seed(0)

    # Parameter configuration aligned with Parameter Settings (Section S1.2)
    gamma, min_samples = 0.85, 2
    theta, eta, epsilon = 0.5, 2.0, 1e-6
    C = 1.0
    K_z = 9
    q = p = 10
    K_h = 1

    X, y = load_mat_dataset(file_path)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # Stratified 80% train / 20% validation split with train-fitted z-score normalization
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=0
    )
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1e-8
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Inject symmetric label noise into training split only
    y_train_noisy = y_train.copy()
    if noise_rate > 0.0:
        n_noisy = int(np.floor(len(y_train_noisy) * noise_rate))
        idx_noise = np.random.choice(len(y_train_noisy), n_noisy, replace=False)
        y_train_noisy[idx_noise] = -y_train_noisy[idx_noise]

    train_data = np.hstack((X_train, y_train_noisy.reshape(-1, 1)))
    gb_details = get_gb_details(train_data, gamma=gamma, min_samples=min_samples)

    S_hat_diag, _, _ = compute_unified_fuzzy_scores(gb_details, theta=theta, eta=eta, epsilon=epsilon)
    X_centers = np.array([gb['center'] for gb in gb_details])
    Y_labels = np.array([gb['label'] for gb in gb_details])

    beta, We, Wh, scaler_params = ddfgb_bls_train(
        X_centers, Y_labels, S_hat_diag, C=C, q=q, K_z=K_z, p=p, K_h=K_h
    )

    A_val = ddfgb_bls_predict(X_val, We, Wh, scaler_params, K_z=K_z, K_h=K_h)
    val_scores = np.dot(A_val, beta).ravel()
    val_preds = np.where(np.sign(val_scores) == 0, 1, np.sign(val_scores))

    val_auc, val_gmean = compute_metrics(y_val, val_preds, val_scores)

    print(f"Dataset: {dataset_name} | Noise: {noise_rate * 100:.0f}% | "
          f"AUC = {val_auc:.4f}, G-mean = {val_gmean:.2f}%")


def main():
    data_dir = './datasets'
    noise_rate = 0.20

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mat_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.mat'))
    if not mat_files:
        print(f"No .mat datasets found in '{data_dir}'. "
              f"Please place binary .mat file(s) ([features, label]) into '{data_dir}' and re-run.")
        return

    for file_name in mat_files:
        run_single_dataset(os.path.join(data_dir, file_name), noise_rate=noise_rate)


if __name__ == '__main__':
    main()
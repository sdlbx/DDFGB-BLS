# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import orth


def tansig(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0


def compute_unified_fuzzy_scores(gb_details, theta=0.5, eta=2.0, epsilon=1e-6):
    M = len(gb_details)
    pos_gbs = [gb for gb in gb_details if gb['label'] == 1]
    neg_gbs = [gb for gb in gb_details if gb['label'] == -1]

    M_plus, M_minus = len(pos_gbs), len(neg_gbs)
    if M_plus == 0 or M_minus == 0:
        return np.ones(M), 1.0, 1.0

    m = min(M_plus, M_minus)
    omega_plus = m / M_plus
    omega_minus = m / M_minus

    # Weighted class centers (Eq. 4)
    def calc_weighted_center(gbs):
        numerator = sum(gb['num'] * gb['purity'] * gb['center'] for gb in gbs)
        denominator = sum(gb['num'] * gb['purity'] for gb in gbs)
        return numerator / denominator

    C_plus = calc_weighted_center(pos_gbs)
    C_minus = calc_weighted_center(neg_gbs)

    D_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            dist_c = np.linalg.norm(gb_details[i]['center'] - gb_details[j]['center'])
            d = max(0.0, dist_c - gb_details[i]['r'] - gb_details[j]['r'])
            D_matrix[i, j] = D_matrix[j, i] = d

    S_hat_diag = np.zeros(M)
    for i, gb in enumerate(gb_details):
        is_pos = (gb['label'] == 1)

        # Local membership u_i and global non-membership v_i (Eqs. 2-6)
        same_idx = [j for j, other in enumerate(gb_details) if other['label'] == gb['label'] and j != i]
        opp_idx = [j for j, other in enumerate(gb_details) if other['label'] != gb['label']]

        k_in = int(np.ceil(np.sqrt(len(same_idx)))) if same_idx else 1
        k_out = int(np.ceil(np.sqrt(len(opp_idx)))) if opp_idx else 1

        D_in_dists = sorted(D_matrix[i, j] for j in same_idx)
        D_out_dists = sorted(D_matrix[i, j] for j in opp_idx)

        D_in_k = D_in_dists[min(k_in - 1, len(D_in_dists) - 1)] if D_in_dists else 0.0
        D_out_k = D_out_dists[min(k_out - 1, len(D_out_dists) - 1)] if D_out_dists else 0.0

        rho_in = 1.0 / (D_in_k + epsilon)
        rho_out = 1.0 / (D_out_k + epsilon)
        u_i = rho_in / (rho_in + rho_out)

        C_own = C_plus if is_pos else C_minus
        C_opp = C_minus if is_pos else C_plus
        P_in = max(0.0, np.linalg.norm(gb['center'] - C_own) - gb['r'])
        P_out = max(0.0, np.linalg.norm(gb['center'] - C_opp) - gb['r'])
        v_i = (P_in + epsilon) / (P_in + P_out + 2.0 * epsilon)

        # Unified fuzzy score (Eqs. 7-9)
        s_i = (theta * (u_i ** eta) + (1.0 - theta) * (1.0 - v_i ** eta)) ** (1.0 / eta)
        omega_i = omega_plus if is_pos else omega_minus
        S_hat_diag[i] = omega_i * s_i

    return S_hat_diag, omega_plus, omega_minus


def ddfgb_bls_train(X_centers, Y_labels, S_hat_diag, C, q, K_z, p, K_h):
    n_samples, n_features = X_centers.shape
    H1 = np.hstack((X_centers, 0.1 * np.ones((n_samples, 1))))

    Z_list, We, scaler_params = [], [], []
    for _ in range(K_z):
        we = 2.0 * np.random.rand(n_features + 1, q) - 1.0
        We.append(we)
        A1 = np.dot(H1, we)

        min_val = np.min(A1, axis=0)
        range_val = np.max(A1, axis=0) - min_val
        range_val[range_val == 0] = 1e-8
        scaler_params.append((min_val, range_val))

        Z_list.append(2.0 * ((A1 - min_val) / range_val) - 1.0)
    Z = np.hstack(Z_list)

    H_list, Wh = [], []
    for _ in range(K_h):
        H2 = np.hstack((Z, 0.1 * np.ones((n_samples, 1))))
        in_dim = K_z * q + 1
        if in_dim >= p:
            wh = orth(2.0 * np.random.rand(in_dim, p) - 1.0)
        else:
            wh = orth((2.0 * np.random.rand(in_dim, p)).T - 1.0).T
        Wh.append(wh)
        H_list.append(tansig(np.dot(H2, wh)))
    H = np.hstack(H_list)

    A = np.hstack((Z, H))
    S_hat = np.diag(S_hat_diag)
    Y_vec = Y_labels.reshape(-1, 1)

    # Closed-form solution (Eq. 17)
    I = np.eye(A.shape[1])
    lhs = (1.0 / C) * I + np.dot(A.T, np.dot(S_hat, A))
    rhs = np.dot(A.T, np.dot(S_hat, Y_vec))
    beta = np.dot(np.linalg.pinv(lhs), rhs)

    return beta, We, Wh, scaler_params


def ddfgb_bls_predict(test_X, We, Wh, scaler_params, K_z, K_h):
    n_samples = test_X.shape[0]
    H1_test = np.hstack((test_X, 0.1 * np.ones((n_samples, 1))))

    Z_test = []
    for i in range(K_z):
        A1_test = np.dot(H1_test, We[i])
        min_val, range_val = scaler_params[i]
        Z_test.append(2.0 * ((A1_test - min_val) / range_val) - 1.0)
    Z_test = np.hstack(Z_test)

    H_test = []
    for i in range(K_h):
        H2_test = np.hstack((Z_test, 0.1 * np.ones((n_samples, 1))))
        H_test.append(tansig(np.dot(H2_test, Wh[i])))
    H_test = np.hstack(H_test)

    return np.hstack((Z_test, H_test))
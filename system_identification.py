import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# True system matrices (from partial_observed_policy_gradient)
A_true = np.array([[1., 1.], [0., 1.]])
B_true = np.array([[0.], [1.]])
C_true = np.array([[1., 0.]])
Sigma_w = 0.5*np.eye(2)
Sigma_v = 0.5*np.array([[1.]])


def collect_data(n_steps, rng=None):
    """Simulate the partially observed system and return (u, y)."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros((2, 1))
    us = []
    ys = []
    for _ in range(n_steps):
        u = rng.standard_normal((1, 1))
        w = la.sqrtm(Sigma_w)@rng.standard_normal((2, 1))
        v = la.sqrtm(Sigma_v)@rng.standard_normal((1, 1))
        y = C_true @ x + v
        us.append(u)
        ys.append(np.array(y))
        x = A_true @ x + B_true @ u + w
    return np.hstack(us), np.hstack(ys)




def kalman_smoother(A, B, C, us, ys):
    """Return smoothed state estimates for a linear system."""
    n = A.shape[0]
    T = us.shape[1]
    x_pred = np.zeros((n, 1))
    P_pred = np.eye(n)
    x_filt = np.zeros((n, T))
    P_filt = np.zeros((n, n, T))
    for t in range(T):
        y_t = ys[:, t:t + 1]
        u_t = us[:, t:t + 1]
        S = C @ P_pred @ C.T + np.array(Sigma_v)
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ (y_t - C @ x_pred)
        P_upd = (np.eye(n) - K @ C) @ P_pred
        x_filt[:, t:t + 1] = x_upd
        P_filt[:, :, t] = P_upd
        x_pred = A @ x_upd + B @ u_t
        P_pred = A @ P_upd @ A.T + np.array(Sigma_w)

    x_smooth = np.zeros_like(x_filt)
    x_smooth[:, -1:] = x_filt[:, -1:]
    for t in range(T - 2, -1, -1):
        P_pred = A @ P_filt[:, :, t] @ A.T + np.array(Sigma_w)
        J = P_filt[:, :, t] @ A.T @ np.linalg.inv(P_pred)
        x_smooth[:, t:t + 1] = x_filt[:, t:t + 1] + J @ (
            x_smooth[:, t + 1:t + 2] - (A @ x_filt[:, t:t + 1] + B @ us[:, t + 1:t + 2]))
    return x_smooth


def identify_system(us, ys, n_iter=10, rng=None):
    """Estimate (A, B, C) via an EM-like procedure."""
    if rng is None:
        rng = np.random.default_rng()
    A_est = np.eye(2) + 0.1 * rng.standard_normal((2, 2))
    B_est = 0.1 * rng.standard_normal((2, 1))
    C_est = rng.standard_normal((1, 2))

    for _ in range(n_iter):
        x_smooth = kalman_smoother(A_est, B_est, C_est, us, ys)
        X = x_smooth[:, :-1]
        U = us[:, :-1]
        X_next = x_smooth[:, 1:]
        AB = X_next @ np.linalg.pinv(np.vstack((X, U)))
        A_est = AB[:, :2]
        B_est = AB[:, 2:].reshape(2, 1)
        C_est = ys @ np.linalg.pinv(x_smooth)
    return A_est, B_est, C_est


def one_step_prediction_error(A_est, B_est, C_est):
    """Return root mean squared prediction error for the given data."""
    ys, us = collect_data(2000)
    n = A_est.shape[0]
    T = us.shape[1]
    x_pred = np.zeros((n, 1))
    P_pred = np.eye(n)
    err = 0.0
    for t in range(T):
        y_t = ys[:, t:t + 1]
        y_pred = C_est @ x_pred
        err += np.sum((y_t - y_pred) ** 2)
        S = C_est @ P_pred @ C_est.T + np.array(Sigma_v)
        K = P_pred @ C_est.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ (y_t - y_pred)
        P_upd = (np.eye(n) - K @ C_est) @ P_pred
        if t < T - 1:
            u_t = us[:, t:t + 1]
            x_pred = A_est @ x_upd + B_est @ u_t
            P_pred = A_est @ P_upd @ A_est.T + np.array(Sigma_w)
    return np.sqrt(err / T)


def run_experiment(sample_sizes, n_trials=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    pred_err = []
    for N in sample_sizes:
        errors = []
        for _ in range(n_trials):
            us, ys = collect_data(N, rng)
            A_hat, B_hat, C_hat = identify_system(us, ys, rng=rng)
            errors.append(one_step_prediction_error(A_hat, B_hat, C_hat))
        pred_err.append(np.mean(errors))
    return np.array(pred_err)


def main():
    sample_sizes = [200, 500, 1000, 10000]
    pred_err = run_experiment(sample_sizes, n_trials=10)

    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, pred_err, 'o-')
    plt.xscale('log')
    plt.xlabel('Number of data points')
    plt.ylabel('One step prediction RMSE')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('identification_errors.png')
    plt.show()

if __name__ == '__main__':
    main()


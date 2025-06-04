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

def fit_arx(us, ys, order=2):
    """Fit a simple ARX model of the given order."""
    p = order
    T = us.shape[1]
    Phi = []
    y_target = []
    for t in range(p, T):
        row = []
        for i in range(1, p + 1):
            row.append(ys[:, t - i])
        for i in range(p):
            row.append(us[:, t - i])
        Phi.append(np.hstack(row))
        y_target.append(ys[:, t])
    Phi = np.vstack(Phi)
    y_target = np.vstack(y_target)
    coeffs, _, _, _ = np.linalg.lstsq(Phi, y_target, rcond=None)
    a = coeffs[:p].reshape(-1)
    b = coeffs[p:].reshape(-1)
    return a, b

def impulse_response(a, b, n_steps):
    """Compute impulse response coefficients for the ARX model."""
    p = len(a)
    r = len(b)
    g = np.zeros(n_steps)
    for t in range(n_steps):
        val = b[t] if t < r else 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                val += a[i - 1] * g[t - i]
        g[t] = val
    return g

def ho_kalman(markov, n):
    """Return (A, B, C) from Markov parameters via Ho-Kalman."""
    H0 = la.hankel(markov[1:n + 1], markov[n:2 * n])
    H1 = la.hankel(markov[2:n + 2], markov[n + 1:2 * n + 1])
    U, S, Vh = np.linalg.svd(H0, full_matrices=False)
    S_sqrt = np.diag(np.sqrt(S[:n]))
    U1 = U[:, :n] @ S_sqrt
    V1 = S_sqrt @ Vh[:n, :]
    A = np.linalg.pinv(U1) @ H1 @ np.linalg.pinv(V1)
    B = V1[:, :1]
    C = U1[:1, :]
    return A, B, C

def identify_system(us, ys, n_iter=10, rng=None, order=2):
    """Identify (A, B, C) using an ARX model and Ho-Kalman realization."""
    arx_order = 20
    a, b = fit_arx(us, ys, order=arx_order)
    markov = impulse_response(a, b, 2 * arx_order + 1)
    A_est, B_est, C_est = ho_kalman(markov, order)
    return A_est, B_est.reshape(-1, 1), C_est.reshape(1, -1)


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


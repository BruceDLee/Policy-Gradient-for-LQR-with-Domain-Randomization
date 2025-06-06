import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# True system matrices (from partial_observed_policy_gradient)
A_true = np.array([[1., 1.], [0., 1.]])
B_true = np.array([[0.], [1.]])
C_true = np.array([[1., 0.]])
Sigma_w = np.eye(2)
Sigma_v = np.array([[1.]])


def lqg_controller(A, B, C):
    """Return a stabilizing LQG feedback gain K and observer gain L."""
    Q = np.eye(A.shape[0])
    R = np.eye(B.shape[1])
    P_f = la.solve_discrete_are(A.T, C.T, Sigma_w, Sigma_v)
    L = A @ P_f @ C.T @ np.linalg.inv(C @ P_f @ C.T + Sigma_v)
    P = la.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, L


def collect_data(n_steps, rng=None, closed_loop=False):
    """Simulate the system and return (u, y)."""
    if rng is None:
        rng = np.random.default_rng()

    if closed_loop:
        K, L = lqg_controller(A_true, B_true, C_true)
        x_hat = np.zeros((2, 1))

    x = np.zeros((2, 1))
    us = []
    ys = []
    for _ in range(n_steps):
        if closed_loop:
            u = K @ x_hat + rng.standard_normal((1, 1))
        else:
            u = rng.standard_normal((1, 1))

        w = la.sqrtm(Sigma_w) @ rng.standard_normal((2, 1))
        v = la.sqrtm(Sigma_v) @ rng.standard_normal((1, 1))
        y = C_true @ x + v
        us.append(u)
        ys.append(np.array(y))
        x = A_true @ x + B_true @ u + w

        if closed_loop:
            x_hat = A_true @ x_hat + B_true @ u + L @ (y - C_true @ x_hat)

    return np.hstack(us), np.hstack(ys)


# def kalman_smoother(A, B, C, us, ys):
#     """Return smoothed state estimates for a linear system."""
#     n = A.shape[0]
#     T = us.shape[1]
#     x_pred = np.zeros((n, 1))
#     P_pred = np.eye(n)
#     x_filt = np.zeros((n, T))
#     P_filt = np.zeros((n, n, T))
#     for t in range(T):
#         y_t = ys[:, t:t + 1]
#         u_t = us[:, t:t + 1]
#         S = C @ P_pred @ C.T + np.array(Sigma_v)
#         K = P_pred @ C.T @ np.linalg.inv(S)
#         x_upd = x_pred + K @ (y_t - C @ x_pred)
#         P_upd = (np.eye(n) - K @ C) @ P_pred
#         x_filt[:, t:t + 1] = x_upd
#         P_filt[:, :, t] = P_upd
#         x_pred = A @ x_upd + B @ u_t
#         P_pred = A @ P_upd @ A.T + np.array(Sigma_w)

#     x_smooth = np.zeros_like(x_filt)
#     x_smooth[:, -1:] = x_filt[:, -1:]
#     for t in range(T - 2, -1, -1):
#         P_pred = A @ P_filt[:, :, t] @ A.T + np.array(Sigma_w)
#         J = P_filt[:, :, t] @ A.T @ np.linalg.inv(P_pred)
#         x_smooth[:, t:t + 1] = x_filt[:, t:t + 1] + J @ (
#             x_smooth[:, t + 1:t + 2] - (A @ x_filt[:, t:t + 1] + B @ us[:, t + 1:t + 2]))
#     return x_smooth

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

# def impulse_response(a, b, n_steps):
#     """Compute impulse response coefficients for the ARX model."""
#     p = len(a)
#     r = len(b)
#     g = np.zeros(n_steps)
#     for t in range(n_steps):
#         val = b[t] if t < r else 0.0
#         for i in range(1, p + 1):
#             if t - i >= 0:
#                 val += a[i - 1] * g[t - i]
#         g[t] = val
#     return g

def block_hankel(data, rows, cols):
    """Return a block Hankel matrix with the given rows and cols."""
    d, T = data.shape
    H = np.zeros((d * rows, cols))
    for i in range(rows):
        H[i * d:(i + 1) * d, :] = data[:, i:i + cols]
    return H


def autoregressive_identification(us, ys, order=2, blocks=10):
    """Subspace identification of a state-space model via autoregressive_identification."""
    m, T = us.shape
    p = ys.shape[0]
    if 2 * blocks >= T:
        raise ValueError("Insufficient data for the chosen block size")

    N = T - 2 * blocks + 1

    
    Up = block_hankel(us[:, :T - blocks], blocks, N)
    # Uf = block_hankel(us[:, blocks:], blocks, N)
    Yp = block_hankel(ys[:, :T - blocks], blocks, N)
    Yf = block_hankel(ys[:, blocks:], 1, N)

    breakpoint()

    W = np.vstack([Up, Yp])
    G = Yf@W.T@la.inv(W@W.T)

    ### TODO! G consists of the Markov parameters for the observer system. 
    # In particular, it maps u{t-1}, ... u_{t-block} y_{t-1} ... y_{t-block} to \hat y_{t+1}
    # By forming the approxpriate Toeplitz matrix and taking the SVD, we should have what looks like
    # an observablity controllability product. Then by truncating to "order" we should be able to immediately
    # read off estimates for C, A, and B


    return A_est, B_est, C_est

def identify_system(us, ys, n_iter=10, rng=None, order=2):
    """Identify (A, B, C) using a subspace (autoregressive_identification) method."""
    block_size = 50
    A_est, B_est, C_est = autoregressive_identification(us, ys, order=order, blocks=block_size)
    return A_est, B_est.reshape(-1, 1), C_est.reshape(1, -1)


def one_step_prediction_error(A_est, B_est, C_est, rng):
    """Return root mean squared prediction error for the given data."""
    us, ys = collect_data(50000, rng, closed_loop=True)
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
            us, ys = collect_data(N, rng, closed_loop=True)
            A_hat, B_hat, C_hat = identify_system(us, ys, rng=rng)
            errors.append(one_step_prediction_error(A_hat, B_hat, C_hat, rng))
        pred_err.append(np.mean(errors))
    return np.array(pred_err)


def main():
    print('best error: ', one_step_prediction_error(A_true, B_true, C_true, np.random.default_rng()))

    sample_sizes = [500]
    pred_err = run_experiment(sample_sizes, n_trials=10)

    
    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, pred_err, 'o-')
    # plt.xscale('log')
    plt.xlabel('Number of data points')
    plt.ylabel('One step prediction RMSE')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('identification_errors.png')
    plt.show()

if __name__ == '__main__':
    main()


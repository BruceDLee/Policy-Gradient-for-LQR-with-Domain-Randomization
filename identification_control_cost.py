import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from system_identification import collect_data, identify_system

# True system matrices
A_true = np.array([[1., 1.], [0., 1.]])
B_true = np.array([[0.], [1.]])
C_true = np.array([[1., 0.]])

# Cost weights
Q_cost = np.eye(2)
R_cost = np.array([[1.]])

# Noise covariances
Sigma_w = np.eye(2)
Sigma_v = np.array([[1.]])


def lqg_controller(A, B, C):
    """Return optimal LQG controller matrices for system (A, B, C)."""
    P_f = la.solve_discrete_are(A.T, C.T, Sigma_w, Sigma_v)
    L = A @ P_f @ C.T @ np.linalg.inv(C @ P_f @ C.T + Sigma_v)
    P = la.solve_discrete_are(A, B, Q_cost, R_cost)
    K = -np.linalg.inv(B.T @ P @ B + R_cost) @ (B.T @ P @ A)
    A_K = A + B @ K - L @ C
    B_K = L
    C_K = K
    return A_K, B_K, C_K


def lqg_cost(A, B, C, A_K, B_K, C_K):
    """Infinite-horizon LQG cost for the given controller."""
    F = np.block([[A, B @ C_K], [B_K @ C, A_K]])
    W = np.block([[Sigma_w, np.zeros((2, 2))],
                  [np.zeros((2, 2)), B_K @ Sigma_v @ B_K.T]])
    if np.max(np.abs(np.linalg.eigvals(F))) >= 1:
        return np.inf
    Sigma = la.solve_discrete_lyapunov(F, W)
    Sigma_x = Sigma[:2, :2]
    Sigma_z = Sigma[2:, 2:]
    return np.trace(Q_cost @ Sigma_x) + np.trace(C_K @ Sigma_z @ C_K.T)


# Cost of the optimal controller for the true system
A_K_opt, B_K_opt, C_K_opt = lqg_controller(A_true, B_true, C_true)
OPTIMAL_COST = lqg_cost(A_true, B_true, C_true, A_K_opt, B_K_opt, C_K_opt)


def run_experiment(sample_sizes, n_trials=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    excess_costs = []
    for N in sample_sizes:
        trial_costs = []
        for _ in range(n_trials):
            us, ys = collect_data(N, rng)
            A_hat, B_hat, C_hat = identify_system(us, ys, rng=rng)
            A_K_hat, B_K_hat, C_K_hat = lqg_controller(A_hat, B_hat, C_hat)
            cost_hat = lqg_cost(A_true, B_true, C_true,
                                A_K_hat, B_K_hat, C_K_hat)
            trial_costs.append(cost_hat - OPTIMAL_COST)
        excess_costs.append(np.mean(trial_costs))
    return np.array(excess_costs)


def main():
    sample_sizes = [20, 50, 100, 200, 500, 1000]
    excess = run_experiment(sample_sizes, n_trials=10)
    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, excess, 'o-')
    plt.xscale('log')
    plt.xlabel('Number of data points')
    plt.ylabel('Excess LQG cost')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('identification_excess_cost.png')
    plt.show()


if __name__ == '__main__':
    main()

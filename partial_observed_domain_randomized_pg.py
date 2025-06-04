import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

# Use identification utilities
import system_identification as si

jax.config.update("jax_enable_x64", True)

"""Domain randomized policy gradient for the partially observed LQR example.

This script first identifies the system dynamics from simulated data and then
runs a policy gradient method with domain randomization.  The implementation
mirrors the fully observed domain randomized algorithm in `stabilization.py`.
"""

# Noise covariances (same as in system_identification and partial_observed_policy_gradient)
Sigma_w = jnp.eye(2)
Sigma_v = jnp.array([[1.]])

# Cost weights
Q_cost = jnp.eye(2)
R_cost = jnp.array([[1.]])

@jax.jit
def dlyap(A_mat, Q_mat):
    """Solve X = A X A^T + Q for X."""
    n = A_mat.shape[0]
    lhs = jnp.eye(n * n) - jnp.kron(A_mat, A_mat)
    x = jnp.linalg.solve(lhs, Q_mat.reshape(-1))
    return x.reshape(n, n)

@jax.jit
def cost(params, As, Bs, Cs):
    """Average infinite-horizon cost over sampled systems."""
    A_K, B_K, C_K = params

    def single_cost(A, B, C):
        F = jnp.block([[A, B @ C_K],
                       [B_K @ C, A_K]])
        W = jnp.block([[Sigma_w, jnp.zeros((2, 2))],
                       [jnp.zeros((2, 2)), B_K @ Sigma_v @ B_K.T]])
        Sigma = dlyap(F, W)
        Sigma_x = Sigma[:2, :2]
        Sigma_z = Sigma[2:, 2:]
        return jnp.trace(Q_cost @ Sigma_x) + jnp.trace(C_K @ Sigma_z @ C_K.T)

    costs = jax.vmap(single_cost, (0, 0, 0))(As, Bs, Cs)
    return jnp.mean(costs)

@jax.jit
def grad_cost(params, As, Bs, Cs):
    return jax.grad(cost)(params, As, Bs, Cs)


def grad_descent(params, As, Bs, Cs, alpha, n_iterations):
    def body_fn(_, p):
        grads = grad_cost(p, As, Bs, Cs)
        return [pp - alpha * gg for pp, gg in zip(p, grads)]

    return jax.lax.fori_loop(0, n_iterations, body_fn, params)


def update_gamma(gamma, As, Bs, Cs, params):
    C_old = cost(params, jnp.sqrt(gamma) * As, jnp.sqrt(gamma) * Bs, Cs)

    gamma_lb = gamma
    gamma_ub = 1.0
    while gamma_ub - gamma_lb > 1e-4:
        gamma_mid = (gamma_ub + gamma_lb) / 2
        C_new = cost(params, jnp.sqrt(gamma_mid) * As, jnp.sqrt(gamma_mid) * Bs, Cs)
        if C_new < 2.5 * C_old:
            gamma_lb = gamma_mid
        elif C_new > 4 * C_old:
            gamma_ub = gamma_mid
        else:
            gamma_lb = gamma_mid
            break
    return gamma_lb


def main():
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------
    # Identification step
    # ------------------------------------------------------------
    us, ys = si.collect_data(200, rng)
    A_hat, B_hat, C_hat = si.identify_system(us, ys, rng=rng)

    # ------------------------------------------------------------
    # Create domain randomization samples around identified model
    # ------------------------------------------------------------
    n_samples = 20
    noise_scale = 0.05
    A_samps = A_hat + noise_scale * rng.standard_normal((n_samples, 2, 2))
    B_samps = B_hat + noise_scale * rng.standard_normal((n_samples, 2, 1))
    C_samps = C_hat + noise_scale * rng.standard_normal((n_samples, 1, 2))

    # Evaluation on the true system
    A_true, B_true, C_true = si.A_true, si.B_true, si.C_true

    # LQG controller based on the identified model for initialization
    P_f = la.solve_discrete_are(A_hat.T, C_hat.T, np.array(Sigma_w), np.array(Sigma_v))
    L_est = A_hat @ P_f @ C_hat.T @ np.linalg.inv(C_hat @ P_f @ C_hat.T + np.array(Sigma_v))
    P = la.solve_discrete_are(A_hat, B_hat, np.array(Q_cost), np.array(R_cost))
    K_lqr_est = -np.linalg.inv(B_hat.T @ P @ B_hat + np.array(R_cost)) @ (B_hat.T @ P @ A_hat)

    params = [
        jnp.array(A_hat + B_hat @ K_lqr_est - L_est @ C_hat),
        jnp.array(L_est),
        jnp.array(K_lqr_est),
    ]

    # ------------------------------------------------------------
    # Domain randomized policy gradient with progressive discounting
    # ------------------------------------------------------------
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(A_samps))).real
    gamma = float(min(0.9 * rho ** (-2), 1.0))

    alpha = 1e-4
    n_iter = 20

    while gamma < 0.999:
        params = grad_descent(
            params,
            jnp.sqrt(gamma) * jnp.array(A_samps),
            jnp.sqrt(gamma) * jnp.array(B_samps),
            jnp.array(C_samps),
            alpha,
            n_iter,
        )
        gamma = float(update_gamma(gamma, jnp.array(A_samps), jnp.array(B_samps), jnp.array(C_samps), params))
        print("gamma updated to", gamma)

    for _ in range(40):
        params = grad_descent(params, jnp.array(A_samps), jnp.array(B_samps), jnp.array(C_samps), alpha, n_iter)

    # ------------------------------------------------------------
    # Evaluate on the true system
    # ------------------------------------------------------------
    final_cost = float(cost(params, jnp.array([A_true]), jnp.array([B_true]), jnp.array([C_true])))
    print("Final cost on true system:", final_cost)


if __name__ == "__main__":
    main()

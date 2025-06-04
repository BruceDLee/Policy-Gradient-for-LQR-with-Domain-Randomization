import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ModuleNotFoundError:
    HAVE_PLT = False

"""Model-based policy gradient for a partially observed LQR system.

We consider
  x_{t+1} = A x_t + B u_t + w_t,
  y_t     = C x_t + v_t,
where w_t ~ N(0, I) and v_t ~ N(0, 1).
The controller has internal state z_t and is parameterized by matrices
(A_K, B_K, C_K):
  z_{t+1} = A_K z_t + B_K y_t,
  u_t     = C_K z_t.
Gradient descent is performed on (A_K, B_K, C_K), and the resulting cost is
compared against the optimal LQG controller obtained from Riccati equations.
"""

# System matrices
A = jnp.array([[1., 1.], [0., 1.]])
B = jnp.array([[0.], [1.]])
C = jnp.array([[1., 0.]])

# Cost weights
Q_cost = jnp.eye(2)
R_cost = jnp.array([[1.]])

# Noise covariances
Sigma_w = jnp.eye(2)
Sigma_v = jnp.array([[1.]])

# Riccati solutions for the optimal LQG controller
P_f = jnp.array(la.solve_discrete_are(A.T, C.T, Sigma_w, Sigma_v))
L_opt = P_f @ C.T @ jnp.linalg.inv(C @ P_f @ C.T + Sigma_v)
P = jnp.array(la.solve_discrete_are(A, B, Q_cost, R_cost))
K_lqr = jnp.linalg.inv(B.T @ P @ B + R_cost) @ (B.T @ P @ A)

# Optimal dynamic controller matrices
A_K_opt = A - B @ K_lqr - L_opt @ C
B_K_opt = L_opt
C_K_opt = -K_lqr

@jax.jit
def dlyap(A_mat, Q_mat):
    """Solve X = A X A^T + Q for X."""
    n = A_mat.shape[0]
    lhs = jnp.eye(n * n) - jnp.kron(A_mat, A_mat)
    x = jnp.linalg.solve(lhs, Q_mat.reshape(-1))
    return x.reshape(n, n)

@jax.jit
def cost(params):
    """Infinite-horizon cost for controller parameters via JAX Lyapunov."""
    A_K, B_K, C_K = params
    F = jnp.block([[A, B @ C_K],
                   [B_K @ C, A_K]])
    W = jnp.block([[Sigma_w, jnp.zeros((2, 2))],
                   [jnp.zeros((2, 2)), B_K @ Sigma_v @ B_K.T]])
    Sigma = dlyap(F, W)
    Sigma_x = Sigma[:2, :2]
    Sigma_z = Sigma[2:, 2:]
    return jnp.trace(Q_cost @ Sigma_x) + jnp.trace(C_K @ Sigma_z @ C_K.T)


def cost_scipy(params):
    """Same cost computed with SciPy's Lyapunov solver for verification."""
    A_K, B_K, C_K = [np.array(p) for p in params]
    F = np.block([[A, B @ C_K],
                  [B_K @ C, A_K]])
    W = np.block([[Sigma_w, np.zeros((2, 2))],
                  [np.zeros((2, 2)), B_K @ Sigma_v @ B_K.T]])
    Sigma = la.solve_discrete_lyapunov(F, W)
    Sigma_x = Sigma[:2, :2]
    Sigma_z = Sigma[2:, 2:]
    return float(np.trace(Q_cost @ Sigma_x) + np.trace(C_K @ Sigma_z @ C_K.T))


def cost_simulation(params, n_steps=10000, burn_in=1000, rng=None):
    """Monte Carlo estimate of the infinite-horizon cost."""
    if rng is None:
        rng = np.random.default_rng()
    A_K, B_K, C_K = [np.array(p) for p in params]
    A_np = np.array(A)
    B_np = np.array(B)
    C_np = np.array(C)
    Q_np = np.array(Q_cost)
    R_np = np.array(R_cost)
    x = np.zeros((2, 1))
    z = np.zeros((2, 1))
    cost_sum = 0.0
    for t in range(n_steps + burn_in):
        u = C_K @ z
        w = rng.standard_normal((2, 1))
        v = rng.standard_normal((1, 1))
        if t >= burn_in:
            cost_sum += float((x.T @ Q_np @ x + u.T @ R_np @ u))
        y = C_np @ x + v
        x = A_np @ x + B_np @ u + w
        z = A_K @ z + B_K @ y
    return cost_sum / n_steps

grad_cost = jax.jit(jax.grad(cost))

alpha = 0.05
n_iterations = 50

params = [0.8 * A_K_opt, 0.8 * B_K_opt, 0.8 * C_K_opt]

cost_history = []
cost_history_scipy = []
for _ in range(n_iterations):
    c = cost(params)
    cost_history.append(float(c))
    cost_history_scipy.append(cost_scipy(params))
    grads = grad_cost(params)
    params = [p - alpha * g for p, g in zip(params, grads)]

cost_history.append(float(cost(params)))
cost_history_scipy.append(cost_scipy(params))
cost_opt = float(cost([A_K_opt, B_K_opt, C_K_opt]))

if HAVE_PLT:
    plt.plot(cost_history, label="learned controller (jax)")
    plt.plot([cost_opt] * len(cost_history), "--", label="optimal controller")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("matplotlib not available, skipping plot")
    print("Final cost:", cost_history[-1])

print("Cost check (JAX vs SciPy) at final step:")
print(cost_history[-1], cost_history_scipy[-1])
opt_scipy = cost_scipy([A_K_opt, B_K_opt, C_K_opt])
print("Optimal cost JAX vs SciPy:")
print(cost_opt, opt_scipy)

print("Monte Carlo check for optimal controller:")
mc_opt = cost_simulation([A_K_opt, B_K_opt, C_K_opt])
print(mc_opt)

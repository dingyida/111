import numpy as np
from scipy.optimize import minimize


def compute_steady_state(lambda_arrivals, mu):
    """
    Compute the steady-state probabilities for the reduced CTMC.
    """
    Lambda = np.sum(lambda_arrivals)
    A = np.sum(lambda_arrivals / (Lambda + mu))
    B = np.sum(lambda_arrivals / (mu * (Lambda + mu)))
    pi0 = (1 - A) / (1 + Lambda * B)
    pi_i0 = pi0 * (lambda_arrivals / ((Lambda + mu) * (1 - A)))
    pi_i1 = pi_i0 * (Lambda / mu)
    return pi0, pi_i0, pi_i1


def compute_aoi_objective(lambda_arrivals, mu, A_mat, g, debug=False):
    """
    Given arrival rates, service rates, observation efficiency,
    and object weights, compute the weighted AoI objective.
    If debug=True, print internal variables.
    """
    # Steady-state probabilities
    pi0, pi_i0, pi_i1 = compute_steady_state(lambda_arrivals, mu)
    if debug:
        print("pi0:", pi0)
        print("pi_i0:", pi_i0)
        print("pi_i1:", pi_i1)

    # Throughput per camera
    phi = mu * (pi_i0 + pi_i1)
    if debug:
        print("phi:", phi)

    # System delay per camera
    Lambda = np.sum(lambda_arrivals)
    pi_total = pi_i0 + pi_i1
    D = np.zeros_like(mu)
    for i in range(len(mu)):
        term1 = pi0 * lambda_arrivals[i] * (1 / mu[i])
        term2 = sum(
            pi_total[k] * lambda_arrivals[i] * (mu[k] / (mu[k] + Lambda)) *
            (1 / (mu[k] + Lambda) + 1 / mu[i])
            for k in range(len(mu))
        )
        D[i] = (term1 + term2) / phi[i]
    if debug:
        print("D:", D)

    # Object update rates
    Phi = A_mat.T.dot(phi)
    if debug:
        print("Phi:", Phi)

    # Corrected AoI calculation
    numerator = A_mat.T.dot(phi * D)
    T = numerator / Phi
    AoI = T + 1 / Phi
    if debug:
        print("T:", T)
        print("AoI:", AoI)

    # Weighted sum
    J = np.dot(g, AoI)
    return J


# Example parameters
np.random.seed(42)
a = 100  # number of cameras
b = 100 # number of objects
mu = np.random.uniform(1.0, 2.0, size=a)
A_mat = np.random.rand(a, b)
g = np.random.rand(b)
Lambda_max = 10

# Debug initial steady-state at x0
x0 = np.ones(a) * Lambda_max / a
print("Debug initial state at x0:")
compute_aoi_objective(x0, mu, A_mat, g, debug=True)

# SLSQP optimization
bounds = [(1e-6, None)] * a
constraints = {'type': 'ineq', 'fun': lambda x: Lambda_max - np.sum(x)}

result = minimize(lambda x: compute_aoi_objective(x, mu, A_mat, g),
                  x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Display results
optimal_lambda = result.x
optimal_objective = result.fun

print("\nOptimal arrival rates (λ_i):", optimal_lambda)
print("Minimum weighted AoI objective J:", optimal_objective)

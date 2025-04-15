import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye

# --- Exact solution u(x, y)
def u(x, y, p, q, a):
    return np.sin(p * np.pi * x / a) * np.sin(q * np.pi * y / a)

# --- Build the 5-point stencil matrix A for interior points
def build_matrix_A(N_inner):
    h = 1.0 / (N_inner + 1)  # grid spacing with boundaries excluded
    factor = 1.0 / (h * h)

    main_diag = 4 * np.ones(N_inner)
    off_diag = -1 * np.ones(N_inner - 1)
    T = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
    I = eye(N_inner)

    A = kron(I, T) + kron(T, I)
    return factor * A

def compute_rhs(N_inner, a, p, q):
    dx = a / (N_inner + 1)
    factor = (np.pi ** 2 / (a * a)) * (p * p + q * q)

    f = np.zeros((N_inner, N_inner))
    for j in range(N_inner):
        for i in range(N_inner):
            x = (i + 1) * dx
            y = (j + 1) * dx
            f[j, i] = factor * u(x, y, p, q, a)

    return f.flatten()
# --- Main convergence analysis
N_values = [4, 8, 16, 32,64, 128]
errors = []

for N in N_values:
    N_inner = N - 2  # Exclude boundary points
    A = build_matrix_A(N_inner)
    b = compute_rhs(N_inner, a=1.0, p=1, q=1)

    # Solve linear system
    u_h = np.linalg.solve(A.toarray(), b)
    
    h = 1.0 / (N_inner + 1)
    x = np.linspace(h, 1 - h, N_inner)
    y = np.linspace(h, 1 - h, N_inner)
    X, Y = np.meshgrid(x, y)
    u_exact_inner = u(X, Y, p=1, q=1, a=1.0).flatten()


    h = 1.0 / (N_inner + 1)
    error = h * np.linalg.norm(u_exact_inner + u_h)


    errors.append(error)


# --- Plot convergence graph
plt.figure()
plt.loglog(N_values, errors, 'o-', label=r'$\|u - u_h\|$')
plt.xlabel("Grid size N")
plt.ylabel("L2 Error")
plt.title("Convergence of Numerical Solution")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("convergence_plot.png")


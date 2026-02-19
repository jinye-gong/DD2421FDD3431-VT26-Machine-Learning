import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# =========================
# Step 1: Set hyperparameters
# =========================
np.random.seed(100)
random.seed(100)

C = 100.0                    # soft-margin regularization
KERNEL_NAME = "RBF"
KERNEL_PARAMS = {"sigma": 2.0}   # for RBF; use {"p": 5} for poly


# =========================
# Step 2: Generate training data
# =========================
classA = np.concatenate((
    np.random.randn(5, 2) * 0.2 + [5, -1],
    np.random.randn(10,  2) * 0.2 + [2, 0.0],
    np.random.randn(20,  2) * 0.2 + [3, 1],
))
classB = np.random.randn(20, 2) * 0.3 + [0.0, -0.5]

X = np.concatenate((classA, classB))
t = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = X.shape[0]
permute = list(range(N))
random.shuffle(permute)
X = X[permute, :]
t = t[permute]


# =========================
# 2) Kernels (choose one)
# =========================
def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel(x, y, p=5):
    return (np.dot(x, y) + 1.0) ** p

def rbf_kernel(x, y, sigma=2.0):
    d = x - y
    return np.exp(-(d @ d) / (2.0 * sigma * sigma))

# Pick kernel here:
KERNEL_NAME = "RBF"
def Kernel(x, y):
    return linear_kernel(x, y)
    # return poly_kernel(x, y, p=5)
    # return rbf_kernel(x, y, sigma=2.0)


# =========================
# Step 4: Precompute P matrix
# =========================
# P_ij = t_i * t_j * K(x_i, x_j)
Kmat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Kmat[i, j] = Kernel(X[i], X[j])

P = (t[:, None] * t[None, :]) * Kmat


# =========================
# Step 5: Define dual objective
# =========================
def objective(alpha):
    # 0.5 * alpha^T P alpha - sum(alpha)
    return 0.5 * (alpha @ P @ alpha) - np.sum(alpha)


# =========================
# Step 6: Add constraints & bounds
# =========================
def zerofun(alpha):
    # sum_i alpha_i t_i = 0
    return np.dot(alpha, t)

#bounds = [(0.0, C) for _ in range(N)]        # soft-margin
bounds = [(0.0, None) for _ in range(N)]   # hard-margin

constraint = {'type': 'eq', 'fun': zerofun}


# =========================
# Step 7: Solve for α
# =========================
start = np.zeros(N)
ret = minimize(
    objective,
    start,
    bounds=bounds,
    constraints=[constraint],
    method='SLSQP'
)

if not ret.success:
    raise ValueError("Optimizer failed: " + str(ret.message))

alpha = ret.x
print("Kernel:", KERNEL_NAME)
print("success:", ret.success)
print("message:", ret.message)


# =========================
# Step 8: Extract support vectors
# =========================
eps = 1e-5
sv_mask = alpha > eps
sv_idx = np.where(sv_mask)[0]

sv_alpha = alpha[sv_idx]
sv_X = X[sv_idx]
sv_t = t[sv_idx]

print("N =", N, " | #support vectors =", len(sv_idx))


# =========================
# Step 9: Compute bias b
# =========================
margin_mask = (sv_alpha < C - eps)  # for soft-margin
margin_idx = sv_idx[margin_mask]

def compute_b():
    # b_s = sum_i alpha_i t_i K(x_s, x_i) - t_s
    if len(margin_idx) > 0:
        idxs = margin_idx
    else:
        idxs = sv_idx

    bs = []
    for s in idxs:
        ssum = 0.0
        for a_i, t_i, x_i in zip(sv_alpha, sv_t, sv_X):
            ssum += a_i * t_i * Kernel(X[s], x_i)
        bs.append(ssum - t[s])
    return float(np.mean(bs))

b = compute_b()
print("b =", b)


# =========================
# Step 10: Indicator + plot
# =========================
def indicator(x):
    ssum = 0.0
    for a_i, t_i, x_i in zip(sv_alpha, sv_t, sv_X):
        ssum += a_i * t_i * Kernel(x, x_i)
    return ssum - b


plt.figure()
plt.plot(classA[:, 0], classA[:, 1], 'b.')
plt.plot(classB[:, 0], classB[:, 1], 'r.')
plt.plot(sv_X[:, 0], sv_X[:, 1], 'ko', markersize=6, fillstyle='none')

plt.axis('equal')

xbound = 5
ybound = 4
xgrid = np.linspace(-xbound, xbound, 200)
ygrid = np.linspace(-ybound, ybound, 200)

grid = np.zeros((len(ygrid), len(xgrid)))
for i, y in enumerate(ygrid):
    for j, x in enumerate(xgrid):
        grid[i, j] = indicator(np.array([x, y]))

plt.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], linewidths=[1, 2, 1])
plt.title(f"SVM ({KERNEL_NAME}), C={C}")
plt.show()

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def simplex_to_xy(p):
    """
    Transforms a 3-dimensional Dirichlet / Multinomial point p = (p1, p2, p3)
    satisfying p1 + p2 + p3 = 1 into 2D Cartesian coordinates on an equilateral triangle.
    Vertices are:
    (0, 0) -> (1, 0, 0)
    (1, 0) -> (0, 1, 0)
    (0.5, sqrt(3)/2) -> (0, 0, 1)
    """
    p = np.asarray(p)
    # x = p2 + 0.5 * p3
    # y = sqrt(3)/2 * p3
    x = p[..., 1] + 0.5 * p[..., 2]
    y = (np.sqrt(3) / 2.0) * p[..., 2]
    return x, y

def plot_dirichlet_contour(alpha, n_grid=200, ax=None, cmap='viridis', levels=20):
    """
    Plots the contour of a Dirichlet distribution with parameter alpha (3-vector)
    on the standard 2-simplex (equilateral triangle).
    """
    if ax is None:
        ax = plt.gca()

    # Generate grid on simplex
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, np.sqrt(3)/2, n_grid)
    X, Y = np.meshgrid(x, y)

    # Invert (x, y) to (p1, p2, p3)
    # p3 = y / (sqrt(3)/2)
    # p2 = x - 0.5 * p3
    # p1 = 1 - p2 - p3
    P3 = Y / (np.sqrt(3) / 2.0)
    P2 = X - 0.5 * P3
    P1 = 1.0 - P2 - P3

    valid = (P1 >= 0) & (P2 >= 0) & (P3 >= 0)
    Z = np.zeros_like(X)

    # Dirichlet pdf = 1/B(alpha) * prod(p_i^{alpha_i - 1})
    log_b = np.sum(sp.gammaln(alpha)) - sp.gammaln(np.sum(alpha))
    
    # Calculate log pdf for valid points
    log_pdf = -log_b + (alpha[0] - 1) * np.log(np.maximum(P1, 1e-12)) \
                     + (alpha[1] - 1) * np.log(np.maximum(P2, 1e-12)) \
                     + (alpha[2] - 1) * np.log(np.maximum(P3, 1e-12))
    
    Z[valid] = np.exp(log_pdf[valid])
    Z[~valid] = np.nan

    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    
    # Draw triangle boundary
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', lw=1.5)
    ax.text(-0.05, -0.05, r'$\mathbf{x}_1$', fontsize=12)
    ax.text(1.02, -0.05, r'$\mathbf{x}_2$', fontsize=12)
    ax.text(0.48, np.sqrt(3)/2 + 0.03, r'$\mathbf{x}_3$', fontsize=12)
    ax.axis('off')
    ax.set_aspect('equal')
    return cs

def plot_gaussian_ellipse(mean, cov, ax=None, n_std=2.0, **kwargs):
    """
    Plots an ellipse representing the covariance of a 2D Gaussian distribution.
    """
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def student_t_pdf(x, mu, lam, nu):
    """
    Univariate Student's t-distribution pdf:
    St(x | mu, lambda, nu) = Gamma(nu/2 + 1/2)/Gamma(nu/2) * (lambda / (pi * nu))^(1/2) * [1 + lambda(x-mu)^2/nu]^(-(nu+1)/2)
    """
    c = sp.gamma((nu + 1) / 2.0) / (sp.gamma(nu / 2.0) * np.sqrt(np.pi * nu / lam))
    return c * (1.0 + (lam * (x - mu)**2) / nu) ** (-(nu + 1) / 2.0)

def von_mises_pdf(theta, mu, kappa):
    """
    Von Mises distribution pdf for periodic variables:
    p(theta | mu, kappa) = 1 / (2*pi*I_0(kappa)) * exp(kappa * cos(theta - mu))
    """
    i0 = sp.i0(kappa)
    return np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * i0)

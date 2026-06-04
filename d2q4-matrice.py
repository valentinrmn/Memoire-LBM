import numpy as np
import matplotlib.pyplot as plt


def rayon_spectral(mu, s2):
    s1 = 2*mu/(mu + 4*nu)
    coeff1 = .5 * (nu/mu)**2*(2*s1**2 + s1**2*s2 + 4*s1*s2) + 1/48 * s2 * (s1**2 - 2*s1) + .25 * nu/mu * (2*s1**2 - 4*s1 + s1**2*s2 -2*s2)
    coeff2 = .5 * (nu/mu)**2*(2*s1**2 + s1**2*s2 + 4*s1*s2) + .25 * nu/mu * (2*s1**2 - 4*s1 + s1**2*s2 -2*s2) + .25 * (s1**2 - 4*s1 + 4 + s1*s2 - 2*s2)
    A = np.array([
        [coeff1, coeff2/2],
        [coeff2/5, coeff1]
    ])
    return np.max(np.abs(np.linalg.eigvals(A)))

nu = 1
mu_max = 10

mu = np.linspace(0.01, mu_max, 500)
s2 = np.linspace(0., 2., 500)

MU, S2 = np.meshgrid(mu, s2)

rho = np.vectorize(rayon_spectral)(MU, S2)
print(f"max $\\rho$ = {rho.max():.4f}")


fig, ax = plt.subplots(figsize=(9,7))
contour = ax.contourf(MU, S2, rho, levels=50, cmap='viridis')
fig.colorbar(contour, ax=ax, label='Rayon spectral')

ax.contour(MU, S2, rho, levels=[1.0], colors='red')
ax.set_xlabel('$\\mu$')
ax.set_ylabel('$s_2$')
plt.tight_layout()
plt.show()

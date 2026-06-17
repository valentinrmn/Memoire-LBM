import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


nu = 1e-1

def rayon_spectral(xi_x, xi_y, mu, s2):
    """
    Renvoie le rayon spectral de la matrice d'amplification du D2Q4.
    """
    s1 = 2*mu/(mu + 4*nu)
    i = complex(0,1)
    Amp = np.array([
        [.5*(np.cos(xi_x) + np.cos(xi_y)), i*(1-s1)*np.sin(xi_x), i*(1-s1)*np.sin(xi_y), .5*(1-s2)*(np.cos(xi_x) - np.cos(xi_y))],
        [.5*i*np.sin(xi_x), (1-s1)*np.cos(xi_x), 0, 0.5*i*(1-s2)*np.sin(xi_x)],
        [0.5*i*np.sin(xi_y), 0, (1-s1)*np.cos(xi_y), 0.5*i*(1-s2)*np.sin(xi_y)],
        [.5 * (np.cos(xi_x) - np.cos(xi_y)), i*(1-s1)*np.sin(xi_x), i*(s1-1)*np.sin(xi_y), .5*(1-s2)*(np.cos(xi_x) + np.cos(xi_y))]])
    return max(np.abs(np.linalg.eigvals(Amp)))


def rayon_spectral2(kx, ky, s1, s2):
    # s1 = 2*mu/(mu + 4**nu)
    i = complex(0,1)
    Amp = np.array([
        [1/2*np.cos(kx) + 1/2*np.cos(ky), -i*s1*np.sin(kx) + i*np.sin(kx), -i*s1*np.sin(ky) + i*np.sin(ky), -1/2*s2*(np.cos(kx) - np.cos(ky)) + 1/2*np.cos(kx) - 1/2*np.cos(ky)],
        [1/2*i*np.sin(kx), -s1*np.cos(kx) + np.cos(kx), 0, -1/2*i*s2*np.sin(kx) + 1/2*i*np.sin(kx)],
        [1/2*i*np.sin(ky), 0, -s1*np.cos(ky) + np.cos(ky), 1/2*i*s2*np.sin(ky) - 1/2*i*np.sin(ky)],
        [1/2*np.cos(kx) - 1/2*np.cos(ky), -i*s1*np.sin(kx) + i*np.sin(kx), i*s1*np.sin(ky) - i*np.sin(ky), -1/2*s2*(np.cos(kx) + np.cos(ky)) + 1/2*np.cos(kx) + 1/2*np.cos(ky)]
    ])
    return max(np.abs(np.linalg.eigvals(Amp)))


s1_init = 1.
s2_init = 1.

N_xi = 100
xi = np.linspace(-np.pi, np.pi, N_xi)
RS = np.zeros((N_xi, N_xi))

for i, xi_x in enumerate(xi):
    for j, xi_y in enumerate(xi):
        RS[i,j] = rayon_spectral2(xi_x, xi_y, s1_init, s2_init)

frequences_axes = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = ['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$']

fig, ax = plt.subplots(figsize=(9,7))
plt.subplots_adjust(bottom=0.15)

mesh = ax.pcolormesh(xi, xi, RS, cmap='viridis')
plt.colorbar(mesh, ax=ax, label='Rayon spectral')

ax.set_xticks(frequences_axes)
ax.set_xticklabels(labels)
ax.set_yticks(frequences_axes)
ax.set_yticklabels(labels)

ax.set_xlabel('$\\xi_x$')
ax.set_ylabel('$\\xi_y$')

ax_s2 = plt.axes( (0.2, 0.05, 0.6, 0.03))
ax_s1 = plt.axes((0.2, 0.02, 0.6, 0.03))
slider_s2 = Slider(ax_s2, '$s_2$', -1., 3., valinit=s2_init, valstep=0.05)
slider_s1 = Slider(ax_s1, '$s_1$', -1., 3., valinit=s1_init, valstep=0.05)

def update(val):
    s2 = slider_s2.val
    s1 = slider_s1.val
    for i, xi_x in enumerate(xi):
        for j, xi_y in enumerate(xi):
            RS[i,j] = rayon_spectral2(xi_x, xi_y, s1, s2)
    mesh.set_array(RS.ravel())
    mesh.set_clim(RS.min(), RS.max())
    fig.canvas.draw_idle()

slider_s2.on_changed(update)
slider_s1.on_changed(update)
plt.show()
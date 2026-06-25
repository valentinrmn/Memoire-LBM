import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


nu = 1e-1

def valeurs_propres(kx, ky, s1, s2):
    mu = 4*nu*s1/(2-s1)
    i = complex(0,1)
    Amp = np.array([
        [1/2*np.cos(kx) + 1/2*np.cos(ky), -i*s1*np.sin(kx) + i*np.sin(kx), -i*s1*np.sin(ky) + i*np.sin(ky), -1/2*s2*(np.cos(kx) - np.cos(ky)) + 1/2*np.cos(kx) - 1/2*np.cos(ky)],
        [1/2*i*np.sin(kx), -s1*np.cos(kx) + np.cos(kx), 0, -1/2*i*s2*np.sin(kx) + 1/2*i*np.sin(kx)],
        [1/2*i*np.sin(ky), 0, -s1*np.cos(ky) + np.cos(ky), 1/2*i*s2*np.sin(ky) - 1/2*i*np.sin(ky)],
        [1/2*np.cos(kx) - 1/2*np.cos(ky), -i*s1*np.sin(kx) + i*np.sin(kx), i*s1*np.sin(ky) - i*np.sin(ky), -1/2*s2*(np.cos(kx) + np.cos(ky)) + 1/2*np.cos(kx) + 1/2*np.cos(ky)]
    ])
    return np.abs(np.linalg.eigvals(Amp))


s1_init = 1.
s2_init = 1.

N_xi = 100
xi = np.linspace(-np.pi, np.pi, N_xi)
lmbda1, lmbda2, lmbda3, lmbda4 = np.zeros((N_xi, N_xi)), np.zeros((N_xi, N_xi)), np.zeros((N_xi, N_xi)), np.zeros((N_xi, N_xi))

for i, xi_x in enumerate(xi):
    for j, xi_y in enumerate(xi):
        eigs = np.sort(valeurs_propres(xi_x, xi_y, s1_init, s2_init))
        lmbda1[i,j], lmbda2[i,j], lmbda3[i,j], lmbda4[i,j] = eigs


# Plot
frequences_axes = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = ['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$']

fig, ax = plt.subplots(2, 2, figsize=(9, 7))
plt.subplots_adjust(bottom=0.17)

ax[0,0].set_title('$\\lambda_1$')
ax[0,1].set_title('$\\lambda_2$')
ax[1,0].set_title('$\\lambda_3$')
ax[1,1].set_title('$\\lambda_4$')

mesh1 = ax[0,0].pcolormesh(xi, xi, lmbda1, cmap='viridis')
mesh2 = ax[0,1].pcolormesh(xi, xi, lmbda2, cmap='viridis')
mesh3 = ax[1,0].pcolormesh(xi, xi, lmbda3, cmap='viridis')
mesh4 = ax[1,1].pcolormesh(xi, xi, lmbda4, cmap='viridis')
plt.colorbar(mesh1, ax=ax[0,0])
plt.colorbar(mesh2, ax=ax[0,1])
plt.colorbar(mesh3, ax=ax[1,0])
plt.colorbar(mesh4, ax=ax[1,1])

for i in range(2):
    for j in range(2):
        ax[i,j].set_xticks(frequences_axes)
        ax[i,j].set_xticklabels(labels)
        ax[i,j].set_yticks(frequences_axes)
        ax[i,j].set_yticklabels(labels)
        ax[i,j].set_xlabel('$\\xi_x$')
        ax[i,j].set_ylabel('$\\xi_y$')

# Sliders

ax_s1 = plt.axes( (0.2, 0.05, 0.6, 0.03))
ax_s2 = plt.axes((0.2, 0.02, 0.6, 0.03))
slider_s1 = Slider(ax_s1, '$s_1$', -1., 3., valinit=s1_init, valstep=0.05)
slider_s2 = Slider(ax_s2, '$s_2$', -1., 3., valinit=s2_init, valstep=0.05)

def update(val):
    s1 = slider_s1.val
    s2 = slider_s2.val
    for i, xi_x in enumerate(xi):
        for j, xi_y in enumerate(xi):
            eigs = np.sort(valeurs_propres(xi_x, xi_y, s1, s2))
            lmbda1[i,j], lmbda2[i,j], lmbda3[i,j], lmbda4[i,j] = eigs

    mesh1.set_array(lmbda1.ravel())
    mesh2.set_array(lmbda2.ravel())
    mesh3.set_array(lmbda3.ravel())
    mesh4.set_array(lmbda4.ravel())

    mesh1.set_clim(lmbda1.min(), lmbda1.max())
    mesh2.set_clim(lmbda2.min(), lmbda2.max())
    mesh3.set_clim(lmbda3.min(), lmbda3.max())
    mesh4.set_clim(lmbda4.min(), lmbda4.max())

    fig.canvas.draw_idle()

slider_s1.on_changed(update)
slider_s2.on_changed(update)
plt.subplots_adjust(bottom=0.20, top=0.92, hspace=0.4, wspace=0.3)
plt.show()
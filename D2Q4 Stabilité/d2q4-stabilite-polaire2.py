import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------------------------
# Différence avec d2q4-stabilite-polaire.py :
#  ici on définit mu en fonction de s1, contrairement à s1 en fonction de mu
#  ça permet de voir la condition de stabilité 0 < s1 < 2, 0 < s2 < 2 plus facilement
# ------------------------------------


nu = 1e-1
def rayon_spectral_polaire(xi, theta, s1, s2):

    xi_x = xi*np.cos(theta)
    xi_y = xi*np.sin(theta)

    # s1 = 2*mu/(mu + 4*nu)
    # s1 = 2 / (1 + 4*nu/mu)
    # indépendant de mu : avant mu n'apparassait qu'à travers s1, mais si s1 est paramètre alors mu disparait complètement du rayon spectral
    i = complex(0,1)
    Amp = np.array([
        [1/2*np.cos(xi_x) + 1/2*np.cos(xi_y), -i*s1*np.sin(xi_x) + i*np.sin(xi_x), -i*s1*np.sin(xi_y) + i*np.sin(xi_y), -1/2*s2*(np.cos(xi_x) - np.cos(xi_y)) + 1/2*np.cos(xi_x) - 1/2*np.cos(xi_y)],
        [1/2*i*np.sin(xi_x), -s1*np.cos(xi_x) + np.cos(xi_x), 0, -1/2*i*s2*np.sin(xi_x) + 1/2*i*np.sin(xi_x)],
        [1/2*i*np.sin(xi_y), 0, -s1*np.cos(xi_y) + np.cos(xi_y), 1/2*i*s2*np.sin(xi_y) - 1/2*i*np.sin(xi_y)],
        [1/2*np.cos(xi_x) - 1/2*np.cos(xi_y), -i*s1*np.sin(xi_x) + i*np.sin(xi_x), i*s1*np.sin(xi_y) - i*np.sin(xi_y), -1/2*s2*(np.cos(xi_x) + np.cos(xi_y)) + 1/2*np.cos(xi_x) + 1/2*np.cos(xi_y)]
    ])
    return max(np.abs(np.linalg.eigvals(Amp)))


s1_init = 1.
s2_init = 1.

N_xi = 100
xi_list = np.linspace(-np.pi, np.pi, N_xi)

angles = [0, np.pi/4, np.pi/2]
angles_labels = ['$0$', '$\\pi/4$', '$\\pi/2$', '$\\pi$']

# pour le plot à la fin
frequences_axes = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels = ['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$']


lines_dict = {}
fig, ax = plt.subplots(figsize=(9,7))
fig.subplots_adjust(bottom=0.20)
for i, (theta, label) in enumerate(zip(angles, angles_labels)):
    rs = [rayon_spectral_polaire(xi, theta, s1_init, s2_init) for xi in xi_list]
    line, = ax.plot(xi_list, rs, label=label)
    lines_dict[i] = line


ax.axhline(1.0, linestyle='--', label='1', color='red')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('Rayon spectral')

ax_s2 = plt.axes( (0.2, 0.05, 0.6, 0.03))
ax_s1 = plt.axes((0.2, 0.02, 0.6, 0.03))
slider_s2 = Slider(ax_s2, '$s_2$', -1., 3., valinit=s2_init, valstep=0.05)
slider_s1 = Slider(ax_s1, '$s_1$', -1., 3., valinit=s1_init, valstep=0.05)

def update(val):
    s2 = slider_s2.val
    s1 = slider_s1.val
    for i, (theta, label) in enumerate(zip(angles, angles_labels)):
        rs = [rayon_spectral_polaire(xi, theta, s1, s2) for xi in xi_list]
        lines_dict[i].set_ydata(rs)
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_s2.on_changed(update)
slider_s1.on_changed(update)

ax.legend()
ax.set_xticks(frequences_axes)
ax.set_xticklabels(labels)
plt.show()
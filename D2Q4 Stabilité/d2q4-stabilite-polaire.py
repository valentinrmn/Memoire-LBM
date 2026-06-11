import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

nu = 1e-1
def rayon_spectral_polaire(xi, theta, mu, s2):

    xi_x = xi*np.cos(theta)
    xi_y = xi*np.sin(theta)

    s1 = 2*mu/(mu + 4*nu)
    i = complex(0,1)
    Amp = np.array([
        [.5*(np.cos(xi_x) + np.cos(xi_y)), i*(1-s1)*np.sin(xi_x), i*(1-s1)*np.sin(xi_y), .5*(1-s2)*(np.cos(xi_x) - np.cos(xi_y))],
        [.5*i*np.sin(xi_x), (1-s1)*np.cos(xi_x), 0, 0.5*i*(1-s2)*np.sin(xi_x)],
        [0.5*i*np.sin(xi_y), 0, (1-s1)*np.cos(xi_y), 0.5*i*(1-s2)*np.sin(xi_y)],
        [.5 * (np.cos(xi_x) - np.cos(xi_y)), i*(1-s1)*np.sin(xi_x), i*(s1-1)*np.sin(xi_y), .5*(1-s2)*(np.cos(xi_x) + np.cos(xi_y))]])
    return max(np.abs(np.linalg.eigvals(Amp)))


mu_init = 1.
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
for i, (theta, label) in enumerate(zip(angles, angles_labels)):
    rs = [rayon_spectral_polaire(xi, theta, mu_init, s2_init) for xi in xi_list]
    line, = ax.plot(xi_list, rs, label=label)
    lines_dict[i] = line


ax.axhline(1.0, linestyle='--', label='1', color='red')
ax.set_xlabel('$\\xi$')
ax.set_ylabel('Rayon spectral')

ax_s2 = plt.axes( (0.2, 0.05, 0.6, 0.03))
ax_mu = plt.axes((0.2, 0.02, 0.6, 0.03))
slider_s2 = Slider(ax_s2, '$s_2$', -1., 3., valinit=s2_init, valstep=0.05)
slider_mu = Slider(ax_mu, '$\\mu$', 0.1, 10., valinit=mu_init, valstep=0.05)

def update(val):
    s2 = slider_s2.val
    mu = slider_mu.val
    for i, (theta, label) in enumerate(zip(angles, angles_labels)):
        rs = [rayon_spectral_polaire(xi, theta, mu, s2) for xi in xi_list]
        lines_dict[i].set_ydata(rs)
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_s2.on_changed(update)
slider_mu.on_changed(update)

ax.legend()
ax.set_xticks(frequences_axes)
ax.set_xticklabels(labels)
plt.show()
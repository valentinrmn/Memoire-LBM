import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

nu = 1e-1
def valeurs_propres_polaire(xi, theta, s1, s2):

    xi_x = xi*np.cos(theta)
    xi_y = xi*np.sin(theta)

    i = complex(0,1)
    Amp = np.array([
        [1/2*np.cos(xi_x) + 1/2*np.cos(xi_y), -i*s1*np.sin(xi_x) + i*np.sin(xi_x), -i*s1*np.sin(xi_y) + i*np.sin(xi_y), -1/2*s2*(np.cos(xi_x) - np.cos(xi_y)) + 1/2*np.cos(xi_x) - 1/2*np.cos(xi_y)],
        [1/2*i*np.sin(xi_x), -s1*np.cos(xi_x) + np.cos(xi_x), 0, -1/2*i*s2*np.sin(xi_x) + 1/2*i*np.sin(xi_x)],
        [1/2*i*np.sin(xi_y), 0, -s1*np.cos(xi_y) + np.cos(xi_y), 1/2*i*s2*np.sin(xi_y) - 1/2*i*np.sin(xi_y)],
        [1/2*np.cos(xi_x) - 1/2*np.cos(xi_y), -i*s1*np.sin(xi_x) + i*np.sin(xi_x), i*s1*np.sin(xi_y) - i*np.sin(xi_y), -1/2*s2*(np.cos(xi_x) + np.cos(xi_y)) + 1/2*np.cos(xi_x) + 1/2*np.cos(xi_y)]
    ])
    return np.sort(np.abs(np.linalg.eigvals(Amp)))


s1_init = 1.
s2_init = 1.
N_xi = 100

angles = [0, np.pi/4, np.pi/2]
angles_labels = ['$0$', '$\\pi/4$', '$\\pi/2$']

# pour le plot à la fin
frequences_axes = [-3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2]
labels = ['$-3\\pi/2$', '$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$']

lines_dict = {}
fig, ax = plt.subplots(2, 2, figsize=(9,7))
fig.subplots_adjust(bottom=0.17)

ax[0,0].set_title('$\\lambda_1$')
ax[0,1].set_title('$\\lambda_2$')
ax[1,0].set_title('$\\lambda_3$')
ax[1,1].set_title('$\\lambda_4$')


for i, (theta, label) in enumerate(zip(angles, angles_labels)):
    # rescaling pour quand theta = pi/4 par ex et la distance est plus grande qu'ailleurs
    xi_max = np.pi / max(abs(np.cos(theta)), abs(np.sin(theta)))
    xi_list = np.linspace(-xi_max, xi_max, N_xi)
    res = np.array([valeurs_propres_polaire(xi, theta, s1_init, s2_init) for xi in xi_list])
    line1, = ax[0,0].plot(xi_list, res[:, 0], label=label)
    line2, = ax[0,1].plot(xi_list, res[:, 1], label=label)
    line3, = ax[1,0].plot(xi_list, res[:, 2], label=label)
    line4, = ax[1,1].plot(xi_list, res[:, 3], label=label)
    lines_dict[(0, 0, i)] = line1
    lines_dict[(0, 1, i)] = line2
    lines_dict[(1, 0, i)] = line3
    lines_dict[(1, 1, i)] = line4

for i in range(2):
    for j in range(2):
        ax[i,j].axhline(1.0, linestyle='--', label='1', color='red')
        ax[i,j].set_xlabel('$\\xi$')
        ax[i,j].set_xticks(frequences_axes)
        ax[i,j].set_xticklabels(labels)
        ax[i,j].set_ylim([0., 1.25])
        ax[i,j].legend()

        

ax_s1 = plt.axes( (0.2, 0.05, 0.6, 0.03))
ax_s2 = plt.axes((0.2, 0.02, 0.6, 0.03))
slider_s1 = Slider(ax_s1, '$s_1$', -1., 3., valinit=s1_init, valstep=0.05)
slider_s2 = Slider(ax_s2, '$s_2$', -1., 3., valinit=s2_init, valstep=0.05)

def update(val):
    s1 = slider_s1.val
    s2 = slider_s2.val
    for i, (theta, label) in enumerate(zip(angles, angles_labels)):
        xi_max = np.pi / max(abs(np.cos(theta)), abs(np.sin(theta)))
        xi_list = np.linspace(-xi_max, xi_max, N_xi)
        res = np.array([valeurs_propres_polaire(xi, theta, s1, s2) for xi in xi_list])
        lines_dict[(0, 0, i)].set_ydata(res[:, 0])
        lines_dict[(0, 1, i)].set_ydata(res[:, 1])
        lines_dict[(1, 0, i)].set_ydata(res[:, 2])
        lines_dict[(1, 1, i)].set_ydata(res[:, 3])

    for i in range(2):
        for j in range(2):
            ax[i,j].autoscale_view()
    fig.canvas.draw_idle()

slider_s1.on_changed(update)
slider_s2.on_changed(update)
plt.subplots_adjust(bottom=0.20, top=0.92, hspace=0.4, wspace=0.3)
plt.show()
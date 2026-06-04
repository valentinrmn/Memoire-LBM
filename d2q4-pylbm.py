import pylbm
import numpy as np
import pylab as plt
import sympy as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable


u, X, Y, LA = sp.symbols('u, X, Y, LA')

# Paramètres du problème
nu = 1e-1


## Solution exacte
def uex(x, y, t, k, l):
    return np.sin(k*np.pi*x)*np.sin(l*np.pi*y)*np.exp(-(k**2+l**2)*np.pi**2*nu*t)


def plot(i, j, z, title):
    im = axarr[i,j].imshow(z)
    divider = make_axes_locatable(axarr[i, j])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, format='%6.0e')
    axarr[i, j].xaxis.set_visible(False)
    axarr[i, j].yaxis.set_visible(False)
    axarr[i, j].set_title(title)

# Domaine
xmin, xmax, ymin, ymax = -1., 1., -1., 1.
Nx = 128
dx = (xmax - xmin)/(Nx+1)
Tf = 1.
mu = 2.
la = mu/dx
s1 = 2. / (1 + 4*nu/mu)
s2 = 1.
k, l = 2, 2

dico_sim = {
    'box': {'x': [xmin, xmax],
            'y': [ymin, ymax],
            'label': 0},
    'space_step': dx,
    'scheme_velocity':la,
    'schemes': [
        {
            'velocities': [1,2,3,4],
            'conserved_moments': u,
            'polynomials': [1, X/LA, Y/LA, (X**2 - Y**2)/(LA**2)],
            'equilibrium': [u, 0., 0., 0.],
            'relaxation_parameters': [0., s1, s1, s2]
        }
     ],
     'init': {u: (uex, (0., k, l))},
     'boundary_conditions': {
         0: {'method': {0: pylbm.bc.AntiBounceBack,}},
     },
     'generator': 'cython',
     'parameters': {LA: la},
}

sol = pylbm.Simulation(dico_sim)
x = sol.domain.x
y = sol.domain.y

f, axarr = plt.subplots(2, 2, figsize=(9,7))
f.suptitle('Heat equation pylbm', fontsize=20)

plot(0, 0, sol.m[u].copy(), 'initial')

while sol.t < Tf:
    sol.one_time_step()

sol.f2m()
z = sol.m[u]
ze = uex(x[:, np.newaxis], y[np.newaxis, :], sol.t, k, l)
plot(1, 0, z, 'final')
plot(0, 1, ze, 'exact')
plot(1, 1, abs(z-ze), 'error')

plt.show()

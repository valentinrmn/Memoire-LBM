import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import time

comparaison_diffusion = True
comp_visuelle_erreur = False

# ---------------------------------------------------------
#          D2Q4 pour l'équation de la chaleur 2D
# \partial_t u = \nu \Delta u, t > 0, x \in [a, b] x [a, b]
#       Étude de l'effet des paramètres sur l'erreur
# ----------------------------------------------------------

xmin, xmax = -1, 1
ymin, ymax = -1, 1

nu = 1e-1

def uex(t, x, y, k=1, l=1):
    return np.sin(k*np.pi*x)*np.sin(l*np.pi*y)*np.exp(-(k**2+l**2)*np.pi**2*nu*t)

def m1eq(m0, a1=0):
    return a1*m0

def m2eq(m0, a2=0):
    return a2*m0

def m3eq(m0, a3=0):
    return a3*m0

def bord_transport(fstar, type='périodique'):
    """
    Conditions de bord (à spécifier) + Transport
    """
    f_new = np.zeros_like(fstar)

    if type == 'dirichlet':
        f_new[0, 0, :] = fstar[-2, 2, :]
        f_new[:, 3, -1] = fstar[:, 1, 1]
        f_new[-1, 2, :] = fstar[1, 0, :]
        f_new[:, 1, 0] = fstar[:, 3, -2]
    elif type == 'périodique':
        f_new[0, 0, :] = fstar[-2, 0, :]
        f_new[:, 3, -1] = fstar[:, 3, 1]
        f_new[-1, 2, :] = fstar[1, 2, :]
        f_new[:, 1, 0] = fstar[:, 1, -2]
    
    # Transport
    f_new[1:, 0, :] = fstar[:-1, 0, :]
    f_new[:, 1, 1:] = fstar[:, 1, :-1]
    f_new[:-1, 2, :] = fstar[1:, 2, :]
    f_new[:, 3, :-1] = fstar[:, 3, 1:]

    return f_new


def simulation(s1, s2, Nx, Ny, T, BC='dirichlet'):

    dx = (xmax-xmin)/(Nx)
    dy = (ymax-ymin)/(Ny)
    x = np.linspace(xmin, xmax, Nx).reshape(-1, 1)
    y = np.linspace(ymin, ymax, Ny).reshape(1, -1)

    mu = 4*nu*s1/(2-s1)

    dt = dx*dx/mu
    Nt = int(T/dt)
    if Nt == 0:
        raise ValueError("Nt=0")

    # Initialisation
    t = 0
    m = np.zeros((Nx, 4, Ny))

    m[:, 0, :] = uex(0, x, y)
    m[:, 1, :] = m1eq(m[:, 0, :])
    m[:, 2, :] = m2eq(m[:, 0, :])
    m[:, 3, :] = m3eq(m[:, 0, :])

    M = np.array([[1, 1, 1, 1],
                  [1, 0, -1, 0],
                  [0, 1, 0, -1],
                  [1, -1, 1, -1],
                  ])
    Minv = np.linalg.inv(M)
    
    for _ in range(Nt):
        m[:, 1, :] = m[:, 1, :] + s1 * (m1eq(m[:, 0, :]) - m[:, 1, :])
        m[:, 2, :] = m[:, 2, :] + s1 * (m2eq(m[:, 0, :]) - m[:, 2, :])
        m[:, 3, :] = m[:, 3, :] + s2 * (m3eq(m[:, 0, :]) - m[:, 3, :])
        # ij, kjl->kil = Minv[i,j] * m[k, j, l]
        fstar = np.einsum('ij, kjl->kil', Minv, m)

        f = bord_transport(fstar, BC)

        m = np.einsum('ij, kjl->kil', M, f)

        t += dt

    return dt, t, x[:, 0], y[0, :], m[:, 0,:]


mu_list = np.linspace(1, 20, 8)#np.array([1, .5, 2, 3.2])
Nx_list = np.array([16, 32, 64, 128])
dx_list = (xmax-xmin)/Nx_list

mu_list = np.sort(mu_list)
Nx_list = np.sort(Nx_list)
dx_list = np.sort(dx_list)

err = np.zeros((mu_list.size, Nx_list.size))

BC = 'périodique'
T = 0.1
s1 = 1.9
s2_opt = 2*(s1**2 -2*s1)/(s1**2-6*s1+4)
s2_list = [1., s2_opt]

for k, s2 in enumerate(s2_list):
    for l, Nx in enumerate(Nx_list):
        dx = dx_list[l]
        dt, t, x, y, m0 = simulation(s1, s2, Nx, Nx, T, BC)
        u_ex = uex(t, x, y)

        err[k, l] = np.sqrt( dx*dx*np.sum((m0 - u_ex)**2) )

fig2, ax2 = plt.subplots(figsize=(9,7))
for k, s2 in enumerate(s2_list):
    if s2 == s2_opt:
        ax2.plot(dx_list, err[k, :], '-o', label=f'$s_2=s_2^*$')
    else:
        ax2.plot(dx_list, err[k, :], '-o', label=f'$s_2=${s2}')
ax2.plot(dx_list, dx_list, '--', label='$O(\\Delta x)$', c='crimson')
ax2.plot(dx_list, dx_list**2, '--', label='$O(\\Delta x^2)$', c='black')
ax2.legend()
ax2.loglog()
ax2.set_xlabel('$\\Delta x$')
ax2.set_ylabel('Erreur $L^2$')
plt.show()





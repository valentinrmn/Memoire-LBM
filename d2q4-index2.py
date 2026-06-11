import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

comparaison_diffusion = False
comp_visuelle_erreur = True

# ---------------------------------------------------------
#          D2Q4 pour l'équation de la chaleur 2D
# \partial_t u = \nu \Delta u, t > 0, x \in [a, b] x [a, b]
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
        # f_new[0,  0, :] = -fstar[0,  2, :]
        # f_new[-1, 2, :] = -fstar[-1, 0, :]
        # f_new[:, 1, -1] = -fstar[:, 3, -1]
        # f_new[:, 3,  0] = -fstar[:, 1,  0]
        f_new[0, 0, :] = fstar[-2, 2, :]
        f_new[:, 3, -1] = fstar[:, 1, 1]
        f_new[-1, 2, :] = fstar[1, 0, :]
        f_new[:, 1, 0] = fstar[:, 3, -2]
    elif type == 'périodique':
        # f_new[-1, 0, :] = fstar[1, 0, :]
        # f_new[0, 2, :] = fstar[-2, 2, :]
        # f_new[:, 1, 0] = fstar[:, 1, -2]
        # f_new[:, 3, -1] = fstar[:, 3, 1]
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


def simulation(s2, Nx, Ny, T, BC='dirichlet', sauvegarde=False, Ncomp=10):

    mu = 1.5

    dx = (xmax-xmin)/(Nx)
    dy = (ymax-ymin)/(Ny)
    x = np.linspace(xmin, xmax, Nx).reshape(-1, 1)
    y = np.linspace(ymin, ymax, Ny).reshape(1, -1)
    # zeros((Nx, Ny)) pour initialiser: éviter meshgrid

    # la = mu/dx
    dt = dx*dx/mu
    Nt = int(T/dt)
    nt_discret_comp = np.linspace(1, Nt, Ncomp, dtype=int)
    t_list = []

    s1 = 2 / (1 + 4*nu/mu)

    # Initialisation
    t = 0
    m = np.zeros((Nx, 4, Ny))

    m[:, 0, :] = uex(0, x, y)
    m[:, 1, :] = m1eq(m[:, 0, :])
    m[:, 2, :] = m2eq(m[:, 0, :])
    m[:, 3, :] = m3eq(m[:, 0, :])

    sauvegardes = []

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
        
        if sauvegarde:
            if _ in nt_discret_comp:
                sauvegardes.append(m[:, 0, :].copy())
                t_list.append(t)
        t += dt
    if sauvegardes:
        return dt, t, x[:, 0], y[0, :], m[:, 0, :], sauvegardes, t_list
    else:
        return dt, t, x[:, 0], y[0, :], m[:, 0, :]

if comparaison_diffusion:
    Nx, Ny = 128, 128
    T = 1.

    BC = 'périodique'

    Ncomp = 50
    
    dt, t_f, x, y, m0, sauvegardes, t_list = simulation(1., Nx, Ny, T, BC, True, Ncomp)
    L = len(sauvegardes)
    max_ex = np.zeros(L)
    max_appr = np.zeros(L)
    for i, t in enumerate(t_list):
        max_ex[i] = np.max(uex(t, x, y))
        max_appr[i] = np.max(sauvegardes[i])
    
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.plot(t_list, max_ex, '-o', label='Exacte')
    ax.plot(t_list, max_appr, '-o', label='Approchée')
    ax.set_yscale('log')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\max_{x, y}(u(t, x, y))$')
    ax.legend()
    plt.show()

if comp_visuelle_erreur:
    cmap = 'viridis'

    Nx, Ny = 128, 128
    T = 1.
    s2 = 2.5

    BC = 'périodique'

    t0 = time.time()
    dt, t, x, y, u_appr = simulation(s2, Nx, Ny, T, BC, False)
    t1 = time.time()
    data = pd.DataFrame({'Nx' : [Nx], \
                         'Ny' : [Ny], \
                         'Temps (s)' : [round(t1-t0, 2)], \
                         's2' : [s2]})
    print(data)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_init = uex(0, X, Y)
    u_ex = uex(t, X, Y)
    print("x =", x, "\n")
    print("y =", y)

    fig, ax = plt.subplots(2, 2, figsize=(9,7))
    pcm1 = ax[0,0].pcolor(X, Y, u_init, cmap=cmap)
    pcm2 = ax[0,1].pcolor(X, Y, u_ex, cmap=cmap)
    pcm3 = ax[1,0].pcolor(X, Y, u_appr, cmap=cmap)
    pcm4 = ax[1,1].pcolor(X, Y, abs(u_ex-u_appr), cmap=cmap)

    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xlabel('$x$')
            ax[i,j].set_ylabel('$y$')

    ax[0,0].set_title('Donnée initiale')
    ax[0,1].set_title(f'Solution exacte à t={t:.2f}')
    ax[1,0].set_title(f'Solution approchée à t={t:.2f}')
    ax[1,1].set_title(f'Erreur')

    fig.colorbar(pcm1, ax=ax[0,0])
    fig.colorbar(pcm2, ax=ax[0,1])
    fig.colorbar(pcm3, ax=ax[1,0])
    fig.colorbar(pcm4, ax=ax[1,1])

    fig.suptitle(f'$D_2Q_4$ pour $\\partial_t u = \\nu \\Delta u$, $\\nu ={nu}$')
    plt.tight_layout()
    plt.show()


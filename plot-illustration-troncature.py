import numpy as np
import matplotlib.pyplot as plt


def s2(s1):
    return -6*(s1**2-2*s1)/(s1**2 -6*s1+12)

s1 = np.linspace(-0.1, 2.1, 200)
s2_im = s2(s1)

fig, ax = plt.subplots(figsize=(9,7))
ax.plot(s1, s2_im, label='$s_2^*(s_1)$')
ax.set_xlabel('$s_1$')
ax.set_ylabel('$s_2^*$')
ax.set_ylim([-0.5, 2.5])
ax.set_xlim([0, 2])
ax.axhline(2, linestyle='--', color='red')
ax.axhline(0, linestyle='--', color='red')
plt.show()


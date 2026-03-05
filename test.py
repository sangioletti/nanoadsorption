import numpy as np



def rho_i_deplete( rho_m, rho, KD_i ):
    term = (rho_m + rho ) / KD_i + 1
    term_sqrt = np.sqrt( term**2 - 4 * rho * rho_m )
    res = ( term - term_sqrt ) / ( 2 / KD_i )
    return res

def rho_i_langmuir( rho_m, rho, KD_i ):
    res = rho_m * KD_i * rho / ( 1 + rho / KD_i )
    return res

rho_m = 1.0
ratio = {}
for KD_i in [10**(-i) for i in range(5,-2,-1)]:
    res1 = []
    for rho in [10**(-i) for i in range(1,-1,-1)]:
        res1.append( (rho, rho_i_langmuir(rho_m, rho, KD_i ) / rho_i_deplete(rho_m, rho, KD_i )  ) )
    ratio[KD_i] = res1

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,6))
colors = plt.cm.viridis(np.linspace(0,1,len(ratio)))

for idx, KD_i in enumerate(sorted(ratio.keys())):
    d_rho, d_y = zip(*ratio[KD_i])
    ax.plot(d_rho, d_y, marker='o', color=colors[idx], label=f'ratio (KD={KD_i:g})')

ax.set_xscale('log')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\rho_i$')
ax.set_yscale('log')
ax.set_title('Depletion vs Langmuir model')
ax.legend()
plt.tight_layout()
plt.savefig('test.png')
plt.show()









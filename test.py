from adsorption import *
from mpmath import mp

mp.dps = 50

test = MultivalentBinding()
nm = test.nm
nm2 = test.nm2
nm3 = test.nm3
um = 1.0e3 * nm
um2 = um*um
nM = ( test.rhostd )**(-1)
N = 10
a_mono = 0.4 * nm
Ree = np.sqrt( N ) * a_mono

# Test the unbiding_probs method
# Calculate the unbinding probabilities for growing values of KLR, store and plot them
with open( 'unbinding_probs.dat', 'w' ) as f:
    f.write( f'KD (nM) pL pR \n' )
    # This is to test KD from 1 pM to 1 mM
    for KD in np.logspace( -3, 6, 100) * nM:
        K0 = KD**(-1)
        KLR = test.K_LR( h = Ree, N = N, a = a_mono, K0 = K0 )
        pL, pR = test.unbinding_probs( sigma_L = 0.01 / nm2, sigma_R = 0.1 / nm2, K_LR = KLR )
        # Save the results to a file
        f.write( f'{KD/nM} {pL} {pR}\n' )

# Test the unbiding_probs method
# Calculate the unbinding probabilities for growing values of KLR, store and plot them
with open( 'unbinding_probs2.dat', 'w' ) as f:
    f.write( f'h/Ree (nM) pL pR \n' )
    # This is to test KD from 1 pM to 1 mM
    for h in np.linspace(0.1 * Ree, 7 * Ree, 20):
        KD =  1 * nM
        K0 = KD**(-1)
        KLR = mp.mpf( test.K_LR( h = h, N = N, a = a_mono, K0 = K0 ) )
        pL, pR = test.unbinding_probs( sigma_L = 0.01 / nm2, sigma_R = 0.01 / nm2, K_LR = KLR )
        # Save the results to a file
        f.write( f'{h/Ree} {pL} {pR}\n' )

# Test the W_bond method
# Calculate the bonding free energy for growing values of h, store and plot them
with open( 'bond_energy_density.dat', 'w' ) as f:
    f.write( f'h/Ree W \n' )
    for h in np.linspace(0.1 * Ree, 7 * Ree, 20):
        K = ( 1 * nM )**(-1)
        W_bond = test.W_bond( h = h, sigma_L = 0.01 / nm2, 
                             sigma_R = 0.01 / nm2, N = 10, a = a_mono, K0 = 1000.0 * nm3 )
        # Save the results to a file
        f.write( f'{h / Ree} {W_bond}\n' )

# Test the W_steric method
# Calculate the steric free energy for growing values of h, store and plot them
with open( 'steric_energy_density.dat', 'w' ) as f:
    f.write( f'h/Ree W \n' )
    for h in np.linspace(0.1 * Ree, 10 * Ree, 20):
        W_steric = test.W_steric( h = h, sigma_polymer = 1.0 / ( 3.0 * nm )**2, N = 10, a = a_mono )
        # Save the results to a file
        f.write( f'{h / Ree} {W_steric}\n' )

# Test the binding fraction method:
# Calculate the binding fraction for growing values of K_bind, store and plot them
with open( 'binding_fraction.dat', 'w' ) as f:
    f.write( f'KD (nM) vs frac_bound \n' )
    for KD in np.logspace( -3, 6, 100) * nM:
        K_bind = KD**(-1)
        frac_bound = test.calculate_bound_fraction( K_bind = K_bind, NP_conc = 0.1 * nM, 
                                                   cell_conc = 1e-4 * nM, R_NP = 50 * nm, 
                                                   A_cell = 100 * um2, verbose = True )
        # Save the results to a file
        f.write( f'{KD/nM} {frac_bound}\n' )
# Plot the results and save the graph in a file
import matplotlib.pyplot as plt
import numpy as np
plt.plot( np.loadtxt( 'binding_fraction.dat', skiprows=1 )[:,0], np.loadtxt( 'binding_fraction.dat', skiprows=1 )[:,1] )
plt.xscale( 'log' )
plt.xlabel( 'K_bind (nM)' )
plt.ylabel( 'frac_bound' )
plt.savefig( 'binding_fraction.png' )

from adsorption import *
from mpmath import mp

mp.dps = 50 # Represent number with 50 digits precision, necessary 
            # to avoid numerical problems when dealing with differences 
	    # between small numbers generated by exponentials

# This is really just for bookkeeping, basically decide the units to be used 
nm = 1.0 # 1 nm in units of length
nm2 = nm*nm # 1 nm^2 in units of area
nm3 = nm*nm*nm # 1 nm^3 in units of volume
mm = 1e6 * nm # 1 mm in units of length
mm3 = mm**3 # 1 mm^3 in units of volume
L = 1e24 * nm3 # 1 L in units of volume
mL = 1e-3 * L # 1 mL in units of volume
um = 1.0e3 * nm # 1 micrometer
um2 = um*um # 1 um^2 in units of area
kT = 1.0 # 1 kT in units of energy
g = 1.0 # 1 g is units of mass, only for bookkeeping

# Define derived units
L = 1e24 * nm3 # 1 L in units of volume
M = 6.023e23 / L # Avogadro's number 
nM = M * 1e-9 # 1 nM 

#****Dosing particles intravenously to animal******
Npdosing=8e12 / mL # Number of particles in dosing solution per mL [mL-1]
Vdosing=5*0.02 * mL # Dosing volume for each animal [mL] (e.g. 5 mL/kg and animal weight 0.02 kg)
fTzone=0.1 # Fraction of dosed particles that ends up in T zone of animal spleen
VTzone=0.5*0.084 * mL # Volume of spleen Tzone in animal [mL] (Assuming 50% of mouse spleen of volume 0.084 mL)
NP_conc=Npdosing*Vdosing*fTzone/VTzone # Number of particles per mL in Tzone of mouse spleen [mL-1]

# Define the design of the multivalent nanoparticle
def Nmonomers( MW ):
    '''Calculate the number of monomers in a PEG chain of a given molecular weight'''
    return int( MW - 18 ) / 44.0

NmonoLong = Nmonomers( 3400 * g ) # Number of monomers in PEG3400 chain
NmonoShort = Nmonomers( 2000 * g ) # Number of monomers in PEG3400 chain
amono = 0.34 * nm # Monomer size in PEG chain
PEG_max_extension = NmonoLong * amono # Max extension of the PEG chain
KD = 0.10 * nM # Dissociation constant in solution between ligand-receptor
K_bind_0 = KD**(-1) # Binding constant in solution between ligand-receptor
R_NP = 35 * nm # Nanoparticle radius in units of length
v_bind = np.pi * R_NP**2 * PEG_max_extension # Volume of binding site 
N_ligands = 150 # Number of ligands on the nanoparticle
sigma_L = 150.0 / ( 4.0 * np.pi * R_NP**2 ) # surface density of ligands
sigma_P2K = 1.0 / ( 2.0 * nm )**2  # Surface density of short PEG chains

# Define system parameters - this info should be checked, for now it is just
# a search from chatGPT
N_lympho = 7.5e7 # Average number of lymphocytes in mouse spleen
N_T_cells = 0.25 * N_lympho # Number of T cells in mouse spleen
V_spleen = 100 * mm3 # Volume of mouse spleen
cell_conc = N_T_cells / V_spleen # T cell concentration in the spleen
A_cell = 100 * um2 # Cell area 
sigma_R_min = 1.0 / um2 # Minimum receptor surface density.
                        # Corresponds to average distance between receptors is 1 um
sigma_R_max = 1e4 / um2 # Maximum receptor surface density
                        # Corresponds to average distance between receptors is 10 nm

# Control output
verbose = False
n_sampling_points = 50 #Number of sampling points for the receptor surface density

# Create the system
system_exact = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort,
                            a_mono = amono, 
                            binding_model = "exact", 
                            polymer_model = "gaussian",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, 
                            )
system_exact_Flory = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort,
                            a_mono = amono, 
                            binding_model = "exact", 
                            polymer_model = "Flory",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, 
                            )
system_saddle = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort, 
                            a_mono = amono, 
                            binding_model = "saddle", 
                            polymer_model = "gaussian",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, )
 
system_fixed_geo_correct = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort, 
                            a_mono = amono, 
                            binding_model = "fixed_geo_correct", 
                            polymer_model = "gaussian",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, )
system_fixed_geo_average = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort, 
                            a_mono = amono, 
                            binding_model = "fixed_geo_average", 
                            polymer_model = "gaussian",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc,) 
system_fixed_geo_lennart = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort, 
                            a_mono = amono, 
                            binding_model = "fixed_geo_lennart", 
                            polymer_model = "brush",
                            #polymer_model = "flory",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, 
                            )
system_fixed_geo_lennart_gaussian = MultivalentBinding( kT=kT, R_NP = R_NP, N_long = NmonoLong , 
                            N_short = NmonoShort, 
                            a_mono = amono, 
                            binding_model = "fixed_geo_shaw", 
                            polymer_model = "gaussian",
                            A_cell = A_cell, 
                            NP_conc = NP_conc, 
                            cell_conc = cell_conc, 
                            )

# Store everything in a file, so we can plot it later
with open( 'adsorption.dat', 'w+' ) as f:
    min_exp = np.log10( sigma_R_min )
    max_exp = np.log10( sigma_R_max )
    M_conc = (A_cell/(np.pi * R_NP**2)) * cell_conc # Concentration of binding sites for NPs
    print( f'NP concentration {NP_conc/(1/mL):5.3e} (1/mL)' )
    print( f'Binding sites concentration {M_conc/(1/mL):5.3e} (1/mL)' )
    print( f'Max adsorbed fraction achievable: min(1, M_conc/NP_conc) = {min(1, M_conc/NP_conc):5.3e}' )
    f.write( f'sigma_R (um^-2) KD_eff (M) adsorbed_fraction\n' )
    for sigma_R in np.logspace( min_exp, max_exp, n_sampling_points ):
        K_bind = system_exact.calculate_binding_constant( sigma_L = sigma_L, 
                                                    sigma_polymer = sigma_L, 
                                                    sigma_R=sigma_R, 
                                                    K_bind_0=K_bind_0, 
                                                    verbose=verbose 
                                                    )
        print( "##################")
        print( "##################")
        K_bind_Flory = system_exact_Flory.calculate_binding_constant( sigma_L = sigma_L, 
                                                    sigma_polymer = sigma_L, 
                                                    sigma_R=sigma_R, 
                                                    K_bind_0=K_bind_0, 
                                                    verbose=verbose 
                                                    )
        print( "##################")
        print( "##################")
        K_bind_saddle = system_saddle.calculate_binding_constant( 
                                                          sigma_L = sigma_L, 
                                                          sigma_polymer = sigma_L, 
                                                          sigma_R = sigma_R, 
                                                          K_bind_0 = K_bind_0, 
                                                          verbose = verbose)
        
        print( "##################")
        print( "##################")
        K_bind_fixed_geo_correct = system_fixed_geo_correct.calculate_binding_constant( 
                                                          sigma_L = sigma_L, 
                                                          sigma_polymer = sigma_L, 
                                                          sigma_R = sigma_R, 
                                                          K_bind_0 = K_bind_0, 
                                                          verbose = verbose)
        print( "##################")
        print( "##################")
        K_bind_fixed_geo_average = system_fixed_geo_average.calculate_binding_constant( 
                                                          sigma_L = sigma_L, 
                                                          sigma_polymer = sigma_L, 
                                                          sigma_R = sigma_R, 
                                                          K_bind_0 = K_bind_0, 
                                                          verbose = verbose)
        
        print( "##################")
        print( "##################")
        K_bind_fixed_geo_lennart = system_fixed_geo_lennart.calculate_binding_constant( 
                                                          sigma_L = sigma_L, 
                                                          sigma_polymer = sigma_L, 
                                                          sigma_R = sigma_R, 
                                                          K_bind_0 = K_bind_0, 
                                                          verbose = verbose)
        print( "##################")
        print( "##################")
        K_bind_fixed_geo_lennart_gaussian = system_fixed_geo_lennart_gaussian.calculate_binding_constant( 
                                                          sigma_L = sigma_L, 
                                                          sigma_polymer = sigma_L, 
                                                          sigma_R = sigma_R, 
                                                          K_bind_0 = K_bind_0, 
                                                          verbose = verbose)
        print( "------------------" )
        print( "------------------" )
        print( f'Effective dissociation constant, exact + ideal polymer (M): {float((1.0 / K_bind) / M):5.3e}' )
        print( f'Effective dissociation constant, exact + Flory polymer (M): {float((1.0 / K_bind_Flory) / M):5.3e}' )
        print( f'Effective dissociation constant, saddle approx + ideal polymer (M): {float((1.0 / K_bind_saddle.real) / M):5.3e}' )
        print( f'Effective dissociation constant fixed geometry, correct chi (M): {float((1.0 / K_bind_fixed_geo_correct) / M):5.3e}' )
        print( f'Effective dissociation constant fixed geometry, average bond length (M): {float((1.0 / K_bind_fixed_geo_average) / M):5.3e}' )
        print( f'Effective dissociation constant fixed geometry, Lennart (Shaw+brush ) (M): {float((1.0 / K_bind_fixed_geo_lennart) / M):5.3e}' )
        print( f'Effective dissociation constant fixed geometry, Ideal polymer (Shaw+ideal ) (M): {float((1.0 / K_bind_fixed_geo_lennart_gaussian) / M):5.3e}' )
        # Now we can calculate the adsorbed fraction
        adsorbed_fraction = system_exact.calculate_bound_fraction( K_bind=K_bind )
        adsorbed_fraction_Flory = system_exact_Flory.calculate_bound_fraction( K_bind=K_bind_Flory )
        adsorbed_fraction_saddle = system_saddle.calculate_bound_fraction( K_bind=K_bind_saddle )
        adsorbed_fraction_fixed_geo_correct = system_fixed_geo_correct.calculate_bound_fraction( K_bind = K_bind_fixed_geo_correct )
        adsorbed_fraction_fixed_geo_average = system_fixed_geo_average.calculate_bound_fraction( K_bind = K_bind_fixed_geo_average )
        adsorbed_fraction_fixed_geo_lennart = system_fixed_geo_lennart.calculate_bound_fraction( K_bind = K_bind_fixed_geo_lennart )
        adsorbed_fraction_fixed_geo_lennart_gaussian = system_fixed_geo_lennart_gaussian.calculate_bound_fraction( K_bind = K_bind_fixed_geo_lennart_gaussian )
        # Print the adsorbed fraction
        out1 = float(sigma_R/(1/um2))
        out2 = float((1 / K_bind) / M)
        out3 = float(adsorbed_fraction)
        out4 = float((1 / K_bind_Flory) / M)
        out5 = float(adsorbed_fraction_Flory)
        out6 = float((1 / K_bind_saddle) / M)
        out7 = float(adsorbed_fraction_saddle)
        out8 = float((1 / K_bind_fixed_geo_correct) / M)
        out9 = float(adsorbed_fraction_fixed_geo_correct)
        out10 = float((1 / K_bind_fixed_geo_average) / M)
        out11 = float(adsorbed_fraction_fixed_geo_average)
        out12 = float((1 / K_bind_fixed_geo_lennart) / M)
        out13 = float(adsorbed_fraction_fixed_geo_lennart)
        out14 = float((1 / K_bind_fixed_geo_lennart_gaussian) / M)
        out15 = float(adsorbed_fraction_fixed_geo_lennart_gaussian)
        f.write( f"""{out1:5.3e} {out2:5.3e} {out3:5.3e} {out4:5.3e} {out5:5.3e}  {out6:5.3e} {out7:5.3e} {out8:5.3e} {out9:5.3e}  {out10:5.3e} {out11:5.3e}  {out12:5.3e} {out13:5.3e} {out14:5.3e} {out15:5.3e}   \n""" )
        print( f'Adsorbed fraction, exact, ideal polymer: {out3:5.3e}' )
        print( f'Adsorbed fraction, exact, Flory polymer: {out5:5.3e}' )
        print( f'Adsorbed fraction, saddle, ideal polymer: {out7:5.3e}' )
        print( f'Adsorbed fraction, fixed geometry, correct bond penalty: {out9:5.3e}' )
        print( f'Adsorbed fraction, fixed geometry, using average bond length: {out11:5.3e}' )
        print( f'Adsorbed fraction, fixed geometry, using Xu&Shaw model (brush): {out13:5.3e}' )
        print( f'Adsorbed fraction, fixed geometry, using Xu&Shaw model (ideal): {out15:5.3e}' )
        print( "------------------" )
        print( "------------------" )

# Now we can plot the results
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt( 'adsorption.dat', skiprows=2 )
plt.xscale( 'log' )
plt.yscale( 'linear' )
plt.plot( data[:,0], data[:,2], label='Exact, ideal', linestyle='solid' )
plt.plot( data[:,0], data[:,4], label='Exact, Flory', linestyle='solid' )
plt.plot( data[:,0], data[:,6], label='Saddle approx, ideal', linestyle='solid' )
plt.plot( data[:,0], data[:,8], label='Fixed geometry, mine', linestyle='--' ) 
plt.plot( data[:,0], data[:,10], label='Fixed geometry, mine, average bond', linestyle='--' )
plt.plot( data[:,0], data[:,12], label='Fixed geometry, Xu&Shaw, brush (Lennart)', linestyle='--' )
plt.plot( data[:,0], data[:,14], label='Fixed geometry, Xu&Shaw, gaussian', linestyle='--' )
plt.xlabel( 'Receptor surface density $(\mu$m$^{-2}$)' )
plt.ylabel( 'Adsorbed fraction' )
plt.legend()
plt.savefig( 'adsorption.png' )

# Calculate the value of W_total for a range of h and the maximum density of ligands and receptors
with open( 'W_total.dat', 'w+' ) as f:
    sigma_R = sigma_R_max
    Ree = np.sqrt( NmonoLong ) * amono
    for h in np.linspace( 0.0, PEG_max_extension, 50 ):
        W_total = system_exact.W_total( h=h, sigma_L=sigma_L, sigma_polymer=sigma_L, 
                                     sigma_R=sigma_R, N=NmonoLong, a=amono, K_bind_0=K_bind_0, verbose=verbose )
        out1 = float(h/(1/nm))
        out2 = float(W_total)
        f.write( f'{out1/Ree:5.3e} {out2:5.3e} \n' )

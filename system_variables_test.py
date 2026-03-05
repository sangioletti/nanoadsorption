from units import *
import numpy as np
from adsorption import Nmonomers

V_SPR = 6 * 10**(-5 ) * mL
A_SPR = 1 * mm2 # Cell area 

Npdosing=4e13 / mL # Number of particles in dosing solution per mL [mL-1]
NP_conc=Npdosing


######################################################
# Here we define the DESIGN OF MULTIVALENT NANOPARTICLE
######################################################

R_NP = 35 * nm # Nanoparticle radius in units of length
N_ligands = 150 # Number of ligands on the nanoparticle
sigma_L = 150.0 / ( 4.0 * np.pi * R_NP**2 ) # surface density of ligands
sigma_P2K = 1.0 / ( 2.0 * nm )**2  # Surface density of short PEG chains, inert and NOT functionalised

# We assume the presence of short (inert) PEG chains + additional ligands on the nanoparticles
amono = 0.28 * nm # Monomer size in PEG chain
NmonoLigands = Nmonomers( 3400 * g ) # Number of monomers in the PEG to which ligands are attached
NmonoShort = Nmonomers( 2000 * g ) # Number of monomers in th short, inert PEG chains
PEG2K_ee = np.sqrt( NmonoShort ) * amono # end-to-end distance of the short PEG chain
PEG_ligands_ee = np.sqrt( NmonoLigands ) * amono # end-to-end extension of the PEG to which ligands are attached
PEG_ligands_max_extension = NmonoLigands * amono # Max extension of the PEG to which ligands are attached
# Calculate the binding volume. This is based on Lennart's Lindfors formula. A different approximation
# could be: v_bind = np.pi * R_NP**2 * PEG_ligands_max_extension
A1 = np.pi / 3.0 
A2 = (( R_NP + PEG_ligands_max_extension) ** 2 - ( R_NP + PEG2K_ee) ** 2 ) * ( R_NP + PEG2K_ee ) 
A3 = - 2.0 * R_NP ** 3 * ( 1.0 - (R_NP + PEG2K_ee) / (R_NP + PEG_ligands_max_extension))
v_bind = A1 * ( A2 + A3 ) 

# Define the binding constant for ligand-receptor binding in solution
KD = 10000.0 * nM # Dissociation constant in solution between ligand-receptor
K_bind_0 = KD**(-1) # Binding constant in solution between ligand-receptor

data_polymers = {}
data_polymers['short'] = {"N": NmonoShort, "a": amono, "sigma": sigma_P2K, "name" : "PEG2K" }
data_polymers['ligands'] = {"N": NmonoLigands, "a": amono, "sigma": sigma_L, "name":"ligands" }
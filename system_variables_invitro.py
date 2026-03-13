from units import *
import numpy as np
from adsorption import Nmonomers

V_SPR = 6 * 10**(-5) * mL # Volume of water over SPR chip
A_SPR = 1 * mm2 # Cell area 

#****Dosing particles intravenously to animal******
# These are parameters controlling the dosing of the nanoparticles in a typical experiments
Npdosing_base = 4e11 / mL
NP_conc = Npdosing_base
cell_conc = 1 / V_SPR # concentration of SPR chip 

######################################################
# Here we define the DESIGN OF MULTIVALENT NANOPARTICLE
######################################################

R_NP = 35*nm # is bare Nanoparticle radius in units of length + radius of hard shell
N_ligands = 80 # Number of ligands on the nanoparticle

# We assume the presence of short (inert) PEG chains + additional ligands on the nanoparticles
amono = 0.28 * nm # Monomer size in PEG chain
akuhn = 0.76 * nm
NmonoLigands = Nmonomers( 3400 * g ) # Number of monomers in the PEG to which ligands are attached
print(f"NmonoLigands: {NmonoLigands}")
NmonoShort = Nmonomers( 2000 * g ) # Number of monomers in th short, inert PEG chains
print(f"NmonoShort: {NmonoShort}")

# Define the binding constant for ligand-receptor binding in solution
KD = 150 * nM # Dissociation constant in solution between ligand-receptor
K_bind_0 = KD**(-1) # Binding constant in solution between ligand-receptor

nonspec_interaction = 0.0 # Strength of nonspecific interaction between whole nanoparticle and surface, in units of kT
binder_linear_size = 3.5 * nm # Linear size of the binder, in this case, an antibody
# Receptor dict — sigma_R is set by the driver script during sweeps
receptor = {"name": "default"}

data_polymers = {}
sigma_L = N_ligands / ( 4.0 * np.pi * R_NP**2 ) # surface density of ligands
sigma_P2K = sigma_L * 11.4
data_polymers['short'] = {"N": NmonoShort, "a": amono, "sigma": sigma_P2K, "name" : "PEG2K", 'akuhn' : akuhn }
data_polymers['ligands'] = {"N": NmonoLigands, "a": amono, "sigma": sigma_L, "name":"ligands", 'akuhn' : akuhn, "K_bind_0": K_bind_0, "receptor": receptor, "binder_linear_size": binder_linear_size }

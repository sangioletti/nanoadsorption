from units import *
import numpy as np

# Define parameters describint the target: T cells in the spleen
# This info should be checked, for now it is just a search from chatGPT
N_lympho = 7.5e7 # Average number of lymphocytes in mouse spleen
N_T_cells = 0.25 * N_lympho # Number of T cells in mouse spleen
V_spleen = 100 * mm3 # Volume of mouse spleen
cell_conc = N_T_cells / V_spleen # T cell concentration in the spleen
A_cell = 100 * um2 # Cell area 

#****Dosing particles intravenously to animal******
# These are parameters controlling the dosing of the nanoparticles in a typical experiments
# Provided by Lennart Lindfors
Npdosing=8e12 / mL # Number of particles in dosing solution per mL [mL-1]
Vdosing=5*0.02 * mL # Dosing volume for each animal [mL] (e.g. 5 mL/kg and animal weight 0.02 kg)
fTzone=0.1 # Fraction of dosed particles that ends up in T zone of animal spleen
VTzone=0.5*0.084 * mL # Volume of spleen Tzone in animal [mL] (Assuming 50% of mouse spleen of volume 0.084 mL)
NP_conc=Npdosing*Vdosing*fTzone/VTzone # Number of particles per mL in Tzone of mouse spleen [mL-1]


# Here we define the DESIGN OF MULTIVALENT NANOPARTICLE
# The following are parameters describing the nanoparticle and the ligands i
# We assume the presence of short (inert) PEG chains + additional ligands on the nanoparticles

# This function simply helps calculating the number of monomers in a PEG chain of a given molecular weight
def Nmonomers( MW ):
    '''Calculate the number of monomers in a PEG chain of a given molecular weight'''
    return int( MW - 18 ) / 44.0

amono = 0.28 * nm # Monomer size in PEG chain
NmonoLigands = Nmonomers( 3400 * g ) # Number of monomers in the PEG to which ligands are attached
NmonoShort = Nmonomers( 2000 * g ) # Number of monomers in th short, inert PEG chains
PEG2K = np.sqrt( NmonoShort ) * amono # end-to-end distance of the short PEG chain
PEG_ligands_max_extension = NmonoLigands * amono # Max extension of the PEG to which ligands are attached
KD = 1.0 * nM # Dissociation constant in solution between ligand-receptor
K_bind_0 = KD**(-1) # Binding constant in solution between ligand-receptor
R_NP = 35 * nm # Nanoparticle radius in units of length
A1 = np.pi / 3.0 
A2 = (( R_NP + PEG_ligands_max_extension) ** 2 - ( R_NP + PEG2K) ** 2 ) * ( R_NP + PEG2K ) 
A3 = - 2.0 * R_NP ** 3 * ( 1.0 - (R_NP + PEG2K) / (R_NP + PEG_ligands_max_extension))
v_bind = A1 * ( A2 + A3 ) 
N_ligands = 150 # Number of ligands on the nanoparticle
sigma_L = 150.0 / ( 4.0 * np.pi * R_NP**2 ) # surface density of ligands
sigma_P2K = 1.0 / ( 2.0 * nm )**2  # Surface density of short PEG chains

data_polymers = {}
data_polymers['short'] = {"N": NmonoShort, "a": amono, "sigma": sigma_P2K, "name" : "PEG2K" }
data_polymers['ligands'] = {"N": NmonoLigands, "a": amono, "sigma": sigma_L, "name":"ligands" }

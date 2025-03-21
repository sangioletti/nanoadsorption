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
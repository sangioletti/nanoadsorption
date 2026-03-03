from adsorption import *
from units import *
from system_variables_L3 import *
from mpmath import mp
import matplotlib.pyplot as plt

precision = mp.dps = 50
assert precision >= 30, AssertionError("You need high precision to avoid numerical problem, set to 50")

verbose = False
n_sampling_points = 50

sigma_R_min = 1.0 / um2
sigma_R_max = 1e4 / um2

min_exp = np.log10(sigma_R_min)
max_exp = np.log10(sigma_R_max)
sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

# Npdosing factors: 1/8, 1/4, 1/2, 1, 2, 4, 8
factors = [2**k for k in range(-3, 4)]  # [0.125, 0.25, 0.5, 1, 2, 4, 8]

# Create a reference system to compute K_bind values (independent of NP_conc)
system_ref = MultivalentBinding(kT=kT, R_NP=R_NP,
                                data_polymers=data_polymers,
                                binding_model="exact",
                                polymer_model="gaussian",
                                A_cell=A_cell,
                                NP_conc=NP_conc,
                                cell_conc=cell_conc)

NP_area = system_ref.NP_area
M_conc = (A_cell / NP_area) * cell_conc

max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
max_N_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1

# Compute K_bind for each NR once (this is the expensive part, independent of NP_conc)
print(f"Computing K_bind for NR = 1..{max_N_receptor - 1} (this is the slow step, done only once)")
K_bind_vs_NR = np.empty(max_N_receptor, dtype=object)
K_bind_vs_NR[0] = 0.0
for NR in range(1, max_N_receptor):
    sigma_R_i = NR / NP_area
    K_bind_vs_NR[NR] = system_ref.calculate_binding_constant(K_bind_0, sigma_R_i)
    print(f"  NR {NR}/{max_N_receptor - 1}")
print("K_bind computation done.")

# Also compute K_bind per sigma_R for the output columns
cached_K_bind = np.zeros(n_sampling_points)
for i, sigma_R in enumerate(sigma_R_values):
    cached_K_bind[i] = system_ref.calculate_binding_constant(K_bind_0, sigma_R, verbose=verbose)
    print(f"  K_bind for sigma_R point {i + 1}/{n_sampling_points}")
print("K_bind per sigma_R done.")

# For each Npdosing factor, recompute bound_vs_receptor from cached K_bind values
# then compute the adsorption curve
results = {}

for factor in factors:
    NP_conc_i = NP_conc * factor
    label = f"Npdosing x{factor:g}"
    print(f"\n--- {label} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

    # Compute bound_vs_receptor using cached K_bind values (depletion model)
    bound_vs_receptor = np.zeros(max_N_receptor)
    for NR in range(1, max_N_receptor):
        K_bind = K_bind_vs_NR[NR]
        if K_bind == np.inf:
            f = 1.0 - mp.exp(-NR)
            max_ads = M_conc * f
            bound_vs_receptor[NR] = min(1.0, max_ads / NP_conc_i)
        else:
            term = (NP_conc_i + M_conc) * K_bind + 1
            sqrt_term = mp.sqrt(mp.power(term, 2) - 4 * NP_conc_i * M_conc * K_bind**2)
            NP_M_conc = (term - sqrt_term) / (2 * K_bind)
            bound_vs_receptor[NR] = NP_M_conc / NP_conc_i

    # Compute bound fraction with fluctuations for each sigma_R
    sigma_out = np.zeros(n_sampling_points)
    KD_out = np.zeros(n_sampling_points)
    frac_out = np.zeros(n_sampling_points)
    nads_out = np.zeros(n_sampling_points)

    for i, sigma_R in enumerate(sigma_R_values):
        bound_fraction = system_ref.calculate_bound_fraction_with_fluctuations(
            K_bind_0, sigma_R, bound_vs_receptor, verbose=False)
        sigma_out[i] = float(sigma_R / (1 / um2))
        KD_out[i] = float((1 / cached_K_bind[i]) / M)
        frac_out[i] = float(bound_fraction)
        nads_out[i] = float(bound_fraction) * NP_conc_i

    results[factor] = (sigma_out, KD_out, frac_out, nads_out)

    # Write per-factor data file
    fname = f"adsorption_Npdosing_x{factor:g}.dat"
    with open(fname, 'w') as f:
        for j in range(n_sampling_points):
            f.write(f"{sigma_out[j]:5.3e} {KD_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
    print(f"  Written to {fname}")

# Plot adsorbed fraction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for factor in factors:
    sigma_out, _, frac_out, nads_out = results[factor]
    ax1.plot(sigma_out, frac_out, linestyle='solid', label=f"Npdosing x{factor:g}")
    ax2.plot(sigma_out, nads_out, linestyle='solid', label=f"Npdosing x{factor:g}")

ax1.set_xscale('log')
ax1.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax1.set_ylabel('Adsorbed fraction')
ax1.legend()

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax2.set_ylabel('Number of adsorbed particles (nm$^{-3}$)')
ax2.legend()

plt.tight_layout()
plt.savefig('adsorption_scan_Npdosing.png')
plt.close()
print("\nPlot saved to adsorption_scan_Npdosing.png")

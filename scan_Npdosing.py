from adsorption import *
from units import *
from system_variables_invivo import *
from mpmath import mp
import matplotlib.pyplot as plt

precision = mp.dps = 50
assert precision >= 30, AssertionError("You need high precision to avoid numerical problem, set to 50")

verbose = False
n_sampling_points = 200

sigma_R_min = 1.0 / um2
sigma_R_max = 2000 / um2

min_exp = np.log10(sigma_R_min)
max_exp = np.log10(sigma_R_max)
sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

# Npdosing factors
factors = [10**k for k in range(-5, 2)]

# Create a reference system to compute K_bind values (independent of NP_conc)
system_ref = MultivalentBinding(kT=kT, R_NP=R_NP,
                                data_polymers=data_polymers,
                                binding_model="exact",
                                polymer_model="Flory-exact",
                                A_cell=A_cell,
                                NP_conc=NP_conc,
                                cell_conc=cell_conc,
                                nonspec_interaction=nonspec_interaction)

# Precompute K_bind for each NR once (expensive step, done only once)
max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 50
print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (this is the slow step, done only once)")
K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(K_bind_0, max_n_receptor)
print("K_bind computation done.")

M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc  # Concentration of binding sites for NPs

# For each Npdosing factor, use the self-consistent solver
results = {}

for factor in factors:
    NP_conc_i = NP_conc * factor
    label = f"Npdosing x{factor:g}"
    print(f"\n--- {label} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

    sigma_out = np.zeros(n_sampling_points)
    frac_out = np.zeros(n_sampling_points)
    nads_out = np.zeros(n_sampling_points)

    for i, sigma_R in enumerate(sigma_R_values):
        bound_fraction = system_ref.calculate_bound_fraction(
                                K_bind_0, sigma_R,
                                fluctuations=True, depletion=True,
                                K_bind_vs_receptors=K_bind_vs_NR,
                                max_n_receptor=max_n_receptor,
                                NP_conc=NP_conc_i,
                                rho_m=M_conc,
                                )

        sigma_out[i] = float(sigma_R / (1 / um2))
        frac_out[i] = float(bound_fraction)
        nads_out[i] = float(bound_fraction) * NP_conc_i * VTzone

    results[factor] = (sigma_out, frac_out, nads_out)

    # Write per-factor data file
    fname = f"adsorption_Npdosing_x{factor:g}.dat"
    with open(fname, 'w') as f:
        for j in range(n_sampling_points):
            f.write(f"{sigma_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
    print(f"  Written to {fname}")

# Plot adsorbed fraction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for factor in factors:
    sigma_out, frac_out, nads_out = results[factor]
    ax1.plot(sigma_out, frac_out, linestyle='solid', label=f"Npdosing x{factor:g}")
    ax2.plot(sigma_out, nads_out, linestyle='solid', label=f"Npdosing x{factor:g}")

#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax1.set_ylabel('Adsorbed fraction (of amount initially in VTzone volume)')
ax1.legend()

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Receptor surface density ($\mu$m$^{-2}$)')
ax2.set_ylabel('Total # of adsorbed particles in VTzone volume')
ax2.legend()

plt.tight_layout()
plt.savefig('adsorption_scan_Npdosing.png')
plt.close()
print("\nPlot saved to adsorption_scan_Npdosing.png")

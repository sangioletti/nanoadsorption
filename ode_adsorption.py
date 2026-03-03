import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
kon = 0.001
koff = 0.001
N_max = 10**8
N_bulk = 24*10**6
N0 = 0  # Initial condition: no bound particles

# Time span
t_max = 0.001
t_eval = np.linspace(0, t_max, 1000)

# --- Nonlinear ODE: dN/dt = kon * (N_max - N) * (N_bulk - N) - koff * N ---
def rhs_nonlinear(t, N):
    return kon * (N_max - N) * (N_bulk - N) - koff * N

sol_nonlinear = solve_ivp(rhs_nonlinear, [0, t_max], [N0], t_eval=t_eval, rtol=1e-10, atol=1e-12)

# --- Linear ODE: dN/dt = kon * (N_max - N) * N_bulk - koff * N ---
# Analytical solution:
#   dN/dt = kon * N_bulk * N_max - (kon * N_bulk + koff) * N
#   This is dN/dt = A - B*N  with A = kon*N_bulk*N_max, B = kon*N_bulk + koff
#   Solution: N(t) = (A/B) * (1 - exp(-B*t))
A = kon * N_bulk * N_max
B = kon * N_bulk + koff
N_eq_linear = A / B
N_linear = N_eq_linear * (1 - np.exp(-B * t_eval))

# Steady states for reference
# Nonlinear: kon*(N_max - N_ss)*(N_bulk - N_ss) = koff*N_ss
# This is a quadratic: kon*N_ss^2 - (kon*(N_max+N_bulk) + koff)*N_ss + kon*N_max*N_bulk = 0
a_coeff = kon
b_coeff = -(kon * (N_max + N_bulk) + koff)
c_coeff = kon * N_max * N_bulk
discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
N_eq_nonlinear = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)  # Smaller root (physical)

print(f"Linear steady state:    N_eq = {N_eq_linear:.2f}")
print(f"Nonlinear steady state: N_eq = {N_eq_nonlinear:.2f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(t_eval, sol_nonlinear.y[0], label='Nonlinear (numerical)', linewidth=2)
plt.plot(t_eval, N_linear, label='Linear (analytical)', linewidth=2, linestyle='dashed')
plt.axhline(y=N_eq_nonlinear, color='C0', linestyle='dotted', alpha=0.5, label=f'Nonlinear steady state = {N_eq_nonlinear:.1f}')
plt.axhline(y=N_eq_linear, color='C1', linestyle='dotted', alpha=0.5, label=f'Linear steady state = {N_eq_linear:.1f}')
plt.xlabel('Time')
plt.ylabel('N(t)')
plt.xscale('log')
plt.title(f'kon={kon}, koff={koff}, N_max={N_max}, N_bulk={N_bulk}')
plt.legend()
plt.tight_layout()
plt.savefig('ode_adsorption.png')
plt.close()
print("Plot saved to ode_adsorption.png")

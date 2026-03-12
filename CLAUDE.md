# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mean-field model of multivalent nanoparticle adsorption to cell surfaces. Computes binding constants and adsorbed fractions as a function of receptor density, ligand–receptor affinity, and polymer tether properties. Based on the Varilly et al. (2012) framework with Derjaguin approximation.

## Running

All scripts are standalone Python files run directly:

```bash
python main.py              # Main adsorption calculation (uses system_variables_L3.py)
python test_self_cons.py    # Self-consistent adsorption with receptor fluctuations
python scan_KD.py           # Scan over dissociation constants
python scan_Nligands.py     # Scan over number of ligands
python scan_Npdosing.py     # Scan over NP concentration (dosing)
python gui.py               # Tkinter GUI for editing system_variables.py and running main.py
```

There is no test suite, build system, or package manager. Dependencies: numpy, scipy, mpmath, matplotlib, PIL (for GUI only).

## Architecture

### Core module: `adsorption.py`

Contains the `MultivalentBinding` class — the central physics engine. Key responsibilities:
- **Polymer models** (`polymer_model` parameter): `"gaussian"`, `"Flory"`, `"rod"`, `"cone"`, `"fjc"`, `"brush"`. Each model defines a chain-end distribution P(z) and confinement factor g(h).
- **Binding models** (`binding_model` parameter): `"exact"` (full numerical integration of Derjaguin potential) or `"saddle"` (saddle-point approximation).
- `K_LR(h)`: area-weighted ligand–receptor bond strength at surface separation h. Uses confined chain-end distribution P_z(h)/g(h).
- `W_bond(h)` / `W_steric(h)` / `W_total(h)`: free energy per unit area contributions.
- `calculate_binding_constant()`: integrates W_total over separation via Derjaguin approximation to get K_bind.
- `calculate_bound_fraction()`: converts K_bind to adsorbed fraction, with or without NP depletion.
- `calculate_bound_fraction_with_fluctuations()`: Poisson-averages over receptor number fluctuations per NP footprint.
- `self_consistent_rho()`: Newton/bisection solver for self-consistent binding with multiple receptor counts.

Uses `mpmath` for arbitrary-precision arithmetic (typically 50 digits) to handle exponentials of large/small numbers without precision loss.

### Units: `units.py`

Defines a consistent unit system with nm as the base length unit. All physical quantities are expressed in these units throughout the code. Key conversions: `nm`, `um`, `um2`, `mm`, `L`, `mL`, `M`, `nM`, `kT`.

### System variable files: `system_variables*.py`

Define physical parameters for specific experimental setups:
- `system_variables_invitro.py` — SPR dosing experiment setup 
- `system_variables_invivo.py` — default parameters, in vivo

Each file defines: `R_NP`, `N_ligands`, `sigma_L`, `sigma_P2K`, PEG chain parameters (`amono`, `NmonoLigands`, `NmonoShort`, `a_kuhn`), `KD`, `K_bind_0`, `NP_conc`, `cell_conc`, `A_cell`, and the `data_polymers` dictionary.

### Data flow

1. A system variables file defines the physical system (NP design, concentrations, affinities).
2. A driver script (e.g., `main.py`) creates a `MultivalentBinding` instance and sweeps over receptor densities.
3. Results are written to `.dat` files and plotted to `.png` files.

## Key conventions

- All lengths in nm, energies in kT, concentrations in nm^-3.
- The `data_polymers` dict must always have a `"ligands"` key (the binding polymer) and can have additional inert polymers (e.g., `"short"` for PEG2K).
- Each polymer entry requires: `"N"` (segments), `"a"` (segment length), `"sigma"` (grafting density), `"name"` (identifier) and `a_kuhn` (kuhn segment length). Optional: `"H"` (brush height), `"w"` (excluded volume), `"theta_max"` (cone half-angle).
- Driver scripts use wildcard imports (`from adsorption import *`, `from units import *`, `from system_variables_X import *`).
- `mpmath` precision must be set to >= 30 digits (`mp.dps = 50` is standard).

## Documentation In Complex_models. Implement additional models for the type of polymer

- `MWC.md` — Derivation and usage of the Milner-Witten-Cates parabolic brush model.
- `ROD.md` — Derivation of the swivelling rod and restricted cone models for K_LR.

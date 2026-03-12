# Multi-Ligand Support

The `MultivalentBinding` class supports multiple independent ligand types on the same nanoparticle, each with its own intrinsic binding constant `K_bind_0` and polymer chain parameters.

## How it works

Each ligand-receptor pair is **independent** (non-competing). The bonding free energy `W_bond` is the sum of independent contributions:

```
W_bond_total(h) = sum_i W_bond_i(h, sigma_L_i, sigma_R_i, K_LR_i)
```

where each `W_bond_i` has its own unbinding probabilities `pL_i`, `pR_i` solved independently via the Varilly et al. saddle-point equations. The steric repulsion `W_steric` sums over **all** polymers (ligands + inert) as before.

## Specifying ligands in `data_polymers`

A polymer entry is treated as an **active ligand** if it has `"K_bind_0"` > 0 in its dictionary. Entries without `"K_bind_0"` (or with `"K_bind_0": 0`) are inert (steric-only).

```python
data_polymers = {
    "ligand_A": {
        "N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
        "name": "ligA", "akuhn": akuhn,
        "K_bind_0": KD_A**(-1),        # intrinsic binding constant
    },
    "ligand_B": {
        "N": NmonoLigands_B, "a": amono, "sigma": sigma_L_B,
        "name": "ligB", "akuhn": akuhn,
        "K_bind_0": KD_B**(-1),        # different affinity
    },
    "PEG2K": {
        "N": NmonoShort, "a": amono, "sigma": sigma_P2K,
        "name": "PEG2K", "akuhn": akuhn,
        # no K_bind_0 -> inert, steric-only
    },
}
```

## The `single_receptor` flag

The constructor parameter `single_receptor` (bool, default `True`) controls whether different ligand types share the same receptor or bind distinct receptor types.

### `single_receptor=True` (default)

All ligand types bind different epitopes on the **same** receptor molecule. They share a single `sigma_R` value that is passed externally to functions like `calculate_binding_constant(sigma_R=...)`.

In this mode:
- **No ligand** in `data_polymers` should specify `"sigma_R"` (raises `ValueError` if found).
- Poisson fluctuations use a single NR ~ Poisson(sigma_R * A_excl).
- All ligand types see the same number of receptors.

This is the appropriate model when different ligands bind non-overlapping sites on the same receptor protein, so the number of available receptors is the same for all ligand types.

### `single_receptor=False`

Each ligand type has its **own cognate receptor** at its own density. Each ligand **must** specify `"sigma_R"` in its `data_polymers` entry (raises `ValueError` if missing).

```python
data_polymers = {
    "ligand_A": {
        ...,
        "K_bind_0": KD_A**(-1),
        "sigma_R": 500 / um2,     # receptor type A density
    },
    "ligand_B": {
        ...,
        "K_bind_0": KD_B**(-1),
        "sigma_R": 200 / um2,     # receptor type B density
    },
}
```

In this mode:
- `sigma_R` in function calls is ignored (each ligand reads its own from `data_polymers`).
- Poisson fluctuations are multi-dimensional: each receptor type fluctuates independently as NR_i ~ Poisson(sigma_R_i * A_excl).
- `calculate_K_bind_vs_receptors` builds a multi-dimensional grid over all (NR_1, NR_2, ...) combinations, flattened for the self-consistent solver.

## API changes from single-ligand version

- `K_bind_0` is **no longer** a function parameter. It is always read from `data_polymers[key]["K_bind_0"]`.
- Functions like `calculate_binding_constant(sigma_R=...)`, `calculate_bound_fraction(sigma_R=...)`, etc. only take `sigma_R` when `single_receptor=True`.
- The `single_receptor` flag is set once at construction time.

## Example usage

### Single receptor (default)

```python
from adsorption import MultivalentBinding
from units import *

system = MultivalentBinding(
    kT=kT, R_NP=35*nm,
    data_polymers=data_polymers,
    binding_model="exact",
    polymer_model="gaussian",
    A_cell=100*um2,
    NP_conc=1e10/mL,
    cell_conc=1e5/mL,
    single_receptor=True,     # default
)

K_bind = system.calculate_binding_constant(sigma_R=100/um2)
```

### Multiple receptor types

```python
system = MultivalentBinding(
    kT=kT, R_NP=35*nm,
    data_polymers=data_polymers,   # each ligand has "sigma_R"
    binding_model="exact",
    polymer_model="gaussian",
    A_cell=100*um2,
    NP_conc=1e10/mL,
    cell_conc=1e5/mL,
    single_receptor=False,
)

K_bind = system.calculate_binding_constant()   # sigma_R from data_polymers
```

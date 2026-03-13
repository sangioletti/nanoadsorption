# Multi-Ligand Support

The `MultivalentBinding` class supports multiple independent ligand types on the same nanoparticle, each with its own intrinsic binding constant `K_bind_0` and polymer chain parameters.

## How it works

Each ligand-receptor pair is **independent** (non-competing). The bonding free energy `W_bond` is the sum of independent contributions:

```
W_bond_total(h) = sum_i W_bond_i(h, sigma_L_i, sigma_R_i, K_LR_i)
```

where each `W_bond_i` has its own unbinding probabilities `pL_i`, `pR_i` solved independently via the Varilly et al. saddle-point equations. The steric repulsion `W_steric` sums over **all** polymers (ligands + inert) as before.

## Specifying ligands in `data_polymers`

A polymer entry is treated as an **active ligand** if it has both:
- `"K_bind_0"` > 0 (intrinsic binding constant)
- `"receptor"` dict (specifies which receptor type it binds)

Entries without `"receptor"` are **inert** (steric-only), regardless of `K_bind_0`.

## Receptor dicts

Each active ligand must have a `"receptor"` key containing a dictionary with:
- `"name"` (required): receptor type identifier
- `"sigma_R"` (optional at construction, required before calculation): receptor surface density

**sigma_R always lives in the receptor dict.** It is the single source of truth. To sweep over sigma_R, modify the dict before calling calculation methods:

```python
receptor["sigma_R"] = new_value
K_bind = system.calculate_binding_constant()
```

**Shared receptors**: ligands sharing the same receptor must reference the **same dict object** (not two dicts with the same name). This ensures sigma_R stays consistent when modified:

```python
rec = {"name": "EGFR", "sigma_R": 500 / um2}  # one dict object

data_polymers = {
    "ligand_A": {..., "receptor": rec},   # same object
    "ligand_B": {..., "receptor": rec},   # same object -> shared receptor
}
```

A `ValueError` is raised if two ligands with the same receptor name reference different dict objects.

## Examples

### Single receptor, sigma_R sweep

```python
receptor = {"name": "default"}

data_polymers = {
    "ligand_A": {
        "N": NmonoLigands, "a": amono, "sigma": sigma_L_A,
        "name": "ligA", "akuhn": akuhn,
        "K_bind_0": KD_A**(-1),
        "receptor": receptor,
    },
    "PEG2K": {
        "N": NmonoShort, "a": amono, "sigma": sigma_P2K,
        "name": "PEG2K", "akuhn": akuhn,
        # no "receptor" -> inert
    },
}

system = MultivalentBinding(
    kT=kT, R_NP=35*nm,
    data_polymers=data_polymers,
    binding_model="exact",
    polymer_model="gaussian",
    A_cell=100*um2, NP_conc=1e10/mL, cell_conc=1e5/mL,
)

# Sweep: just modify the receptor dict
for sigma_R in sigma_R_values:
    receptor["sigma_R"] = sigma_R
    K_bind = system.calculate_binding_constant()
```

### Multiple ligands, shared receptor

```python
rec_shared = {"name": "EGFR", "sigma_R": 500 / um2}

data_polymers = {
    "ligand_A": {..., "K_bind_0": KD_A**(-1), "receptor": rec_shared},
    "ligand_B": {..., "K_bind_0": KD_B**(-1), "receptor": rec_shared},
}
```

Both ligands see the same receptor. Updating `rec_shared["sigma_R"]` affects both.

### Different receptor types

```python
rec_EGFR = {"name": "EGFR", "sigma_R": 500 / um2}
rec_HER2 = {"name": "HER2", "sigma_R": 200 / um2}

data_polymers = {
    "ligand_A": {..., "K_bind_0": KD_A**(-1), "receptor": rec_EGFR},
    "ligand_B": {..., "K_bind_0": KD_B**(-1), "receptor": rec_HER2},
}

system = MultivalentBinding(...)
K_bind = system.calculate_binding_constant()  # reads sigma_R from each receptor dict
```

### Mixed case: some ligands share a receptor, others have their own

```python
rec_EGFR = {"name": "EGFR", "sigma_R": 500 / um2}
rec_HER2 = {"name": "HER2", "sigma_R": 200 / um2}

data_polymers = {
    "ligand_A": {..., "K_bind_0": KD_A**(-1), "receptor": rec_EGFR},
    "ligand_B": {..., "K_bind_0": KD_B**(-1), "receptor": rec_EGFR},  # same as A
    "ligand_C": {..., "K_bind_0": KD_C**(-1), "receptor": rec_HER2},  # independent
}
```

Ligands A and B share EGFR receptor fluctuations; ligand C fluctuates independently on HER2.

## Poisson fluctuations

- **Single receptor type**: 1D Poisson averaging over NR ~ Poisson(sigma_R * A_excl).
- **Multiple receptor types**: multi-dimensional Poisson averaging. Each receptor type fluctuates independently: NR_i ~ Poisson(sigma_R_i * A_excl). The K_bind grid is built over the Cartesian product of receptor numbers.

## API summary

- `K_bind_0` is read from `data_polymers[key]["K_bind_0"]`.
- `sigma_R` is read from `data_polymers[key]["receptor"]["sigma_R"]`. No sigma_R parameter in any method.
- To sweep sigma_R, modify `receptor["sigma_R"]` before calling methods.
- `calculate_K_bind_vs_receptors(max_N_receptor)` temporarily sets sigma_R in receptor dicts and restores them:
  - Single receptor type: returns 1D array `K_bind[NR]`.
  - Multiple receptor types: returns `(K_bind_flat, grid_shape, NR_aves, receptor_names)`.

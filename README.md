# nanoadsorption
Repository coding for a mean-field model of nanoparticle adsorption to cells

## Prerequisites

### Install uv (Python Package Manager)

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/

### Install Rust

**macOS, Linux, and WSL:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
Download and run the installer from: https://rustup.rs/

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

## Running the Simulation

### Step 1: Run the Rust Implementation

Navigate to the rust directory and run the simulation:

```bash
cd rust
cargo run --release
```

This will generate `adsorption_rust.dat` in the rust directory.

### Step 2: Copy Rust Results to Top Level

Copy the generated file to the repository root:

```bash
cp adsorption_rust.dat ..
cd ..
```

### Step 3: Run the Python Implementation

Run the Python simulation and generate plots:

```bash
uv run main.py
```

This will:
- Generate `adsorption.dat` (Python results)
- Create `adsorption.png` (plot from Python results)
- Create `adsorption_rust.png` (plot from Rust results)

## Output Files

- `adsorption.dat` - Python simulation results (3 columns: receptor density, effective KD, adsorbed fraction)
- `adsorption_rust.dat` - Rust simulation results (same format)
- `adsorption.png` - Plot of Python simulation results
- `adsorption_rust.png` - Plot of Rust simulation results

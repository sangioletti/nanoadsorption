#![feature(f128)]

mod calculate_exact;
#[allow(dead_code)]
mod multivalent_binding;
#[allow(dead_code)]
mod params;
use core::num::FpCategory;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use multivalent_binding::{BindingModel, MultiValentBinding, PolymerModel};
use params::{KT, M, SystemParameters, UM2};

/// Generate logarithmically spaced values from 10^min_exp to 10^max_exp
fn logspace(min_exp: f128, max_exp: f128, n_points: usize) -> Vec<f128> {
    let mut values = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let exp = min_exp + (max_exp - min_exp) * (i as f128) / ((n_points - 1) as f128);
        values.push(10.0_f128.powf(exp));
    }
    values
}

fn main() {
    let start = Instant::now();

    let params = SystemParameters::default();

    println!("Nanoparticle Adsorption Simulation Parameters");
    println!("==============================================");
    println!("Nanoparticle radius: {:.2} nm", params.r_np as f64);
    println!("Number of ligands: {}", params.n_ligands as f64);
    println!("Ligand surface density: {:.6e} nm^-2", params.sigma_l as f64);
    println!("Cell concentration: {:.6e} mL^-1", params.cell_conc as f64);
    println!("NP concentration: {:.6e} mL^-1", params.np_conc as f64);
    println!("Binding constant K_0: {:.6e}", params.k_bind_0 as f64);
    println!("\nPolymer Data:");
    println!(
        "  Short PEG ({}): N={:.1}, a={:.3} nm",
        params.data_polymers.short.name,
        params.data_polymers.short.n as f64,
        params.data_polymers.short.a as f64
    );
    println!(
        "  Ligand PEG ({}): N={:.1}, a={:.3} nm",
        params.data_polymers.ligands.name,
        params.data_polymers.ligands.n as f64,
        params.data_polymers.ligands.a as f64
    );

    // Create MultivalentBinding system
    let system = MultiValentBinding::new(
        KT,
        params.r_np,
        params.data_polymers.clone(),
        params.a_cell,
        params.np_conc,
        params.cell_conc,
        BindingModel::Exact,
        PolymerModel::Gaussian,
    );

    println!("\n==============================================");
    println!("Testing MultivalentBinding Implementation");
    println!("==============================================");

    // Test r_ee calculation
    let n = system.data_polymers.ligands.n;
    let a = system.data_polymers.ligands.a;
    let r_ee = system.r_ee(n, a);
    println!("End-to-end distance R_ee: {:.3} nm", r_ee as f64);

    // Test with a sample receptor density
    let sigma_r = 1000.0 / UM2; // 1000 receptors per um^2
    println!("\nTesting with receptor density: {:.3e} nm^-2", sigma_r as f64);

    // Calculate binding constant
    println!("\nCalculating binding constant (this may take a moment)...");
    let k_bind = system.calculate_binding_constant(params.k_bind_0, sigma_r, None, false);
    println!("Binding constant K_bind: {:.6e}", k_bind as f64);

    // Calculate bound fraction
    let bound_fraction = system.calculate_bound_fraction(k_bind, false);
    println!("Bound fraction: {:.6}", bound_fraction as f64);

    // Simulation parameters (matching Python main.py)
    let verbose = false;
    let n_sampling_points = 50;
    let sigma_r_min = 1.0 / UM2; // 1 receptor per um^2
    let sigma_r_max = 1e4 / UM2; // 10,000 receptors per um^2

    println!("\n==============================================");
    println!("Running Full Adsorption Simulation");
    println!("==============================================");
    println!("Sampling points: {}", n_sampling_points);
    println!(
        "Receptor density range: {:.1e} to {:.1e} nm^-2",
        sigma_r_min as f64, sigma_r_max as f64
    );
    println!("\nCalculating adsorption data (this may take several minutes)...");

    match run_adsorption_simulation(
        &system,
        params.k_bind_0,
        sigma_r_min,
        sigma_r_max,
        n_sampling_points,
        verbose,
    ) {
        Ok(()) => {
            println!("\n✓ Simulation complete!");
            println!("Results written to: adsorption.dat");
        }
        Err(e) => {
            eprintln!("Error running simulation: {}", e);
        }
    }

    let elapsed = start.elapsed();
    println!("time taken {:.1}", elapsed.as_secs_f64());
}

/// Run adsorption simulation and write results to file
/// Mirrors Python main.py lines 33-54
fn run_adsorption_simulation(
    system: &MultiValentBinding,
    k_bind_0: f128,
    sigma_r_min: f128,
    sigma_r_max: f128,
    n_sampling_points: usize,
    verbose: bool,
) -> std::io::Result<()> {
    let mut file = File::create("adsorption_rust.dat")?;

    let min_exp = sigma_r_min.log10();
    let max_exp = sigma_r_max.log10();

    // Generate logspace values for receptor surface density
    let sigma_r_values = logspace(min_exp, max_exp, n_sampling_points);

    const NO_TO_TEST: usize = 50;

    let iterations = usize::min(NO_TO_TEST, sigma_r_values.len());

    println!("sigma r values {}", sigma_r_values.len());

    for sigma_r in sigma_r_values[0..iterations].iter() {
        // Calculate binding constant
        let k_bind = system.calculate_binding_constant(k_bind_0, *sigma_r, None, verbose);

        process_value(k_bind, "k_bind");

        // Calculate adsorbed fraction
        let adsorbed_fraction = system.calculate_bound_fraction(k_bind, verbose);

        // process_value(adsorbed_fraction, "adsorbed_fraction");

        // Format output values
        let out1 = sigma_r / (1.0 / UM2); // sigma_R in um^-2
        let out2 = (1.0 / k_bind) / M; // KD_eff in M
        let out3 = adsorbed_fraction;

        // Write to file
        writeln!(file, "{:5.3e} {:5.3e} {:5.3e}", out1 as f64, out2 as f64, out3 as f64)?;
    }

    Ok(())
}

fn process_value(v: f128, label: &str) {
    match v.classify() {
        FpCategory::Normal => {
            println!("Normal value of{}: {}", label, v as f64);
        }
        FpCategory::Infinite => {
            println!("Infinite value of {}: {}", label, v as f64);
            // Handle infinity, e.g., by panicking or returning an error
        }
        FpCategory::Nan => {
            println!("NaN encountered of {}!", label);
            // Handle NaN, e.g., by panicking or returning an error
        }
        FpCategory::Zero => {
            println!("zero value of{}: {}", label, v as f64);
        }
        FpCategory::Subnormal => {
            println!("subnormal value of{}: {}", label, v as f64);
        }
    }
}

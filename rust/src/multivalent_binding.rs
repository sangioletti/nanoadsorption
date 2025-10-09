use std::f128::consts::PI;

use crate::params::{PolymerData, PolymersData};

/// Polymer model types for calculating chain statistics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolymerModel {
    Gaussian,
    Flory,
    SelfAvoiding,
}

/// Binding model types for calculating binding constants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BindingModel {
    Exact,
    Saddle,
}

/// MultivalentBinding struct implementing the mean-field model for
/// nanoparticle adsorption to cells via multivalent binding interactions
#[derive(Debug, Clone)]
pub struct MultiValentBinding {
    /// Polymer data (ligands and short PEG chains)
    pub data_polymers: PolymersData,

    /// Thermal energy (kB*T)
    pub kt: f128,

    /// Units of length (nm)
    pub nm: f128,

    /// Units of area (nm^2)
    pub nm2: f128,

    /// Units of volume (nm^3)
    pub nm3: f128,

    /// Standard density (6.023e23 / (1e24 * nm^3))
    pub rhostd: f128,

    /// Nanoparticle radius
    pub r_np: f128,

    /// Polymer model type (gaussian, flory, self_avoiding)
    pub polymer_model: PolymerModel,

    /// Binding model type (exact, saddle)
    pub binding_model: BindingModel,

    /// Cell surface area
    pub a_cell: f128,

    /// Nanoparticle concentration
    pub np_conc: f128,

    /// Cell concentration
    pub cell_conc: f128,
}

impl MultiValentBinding {
    /// Create a new MultivalentBinding instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kt: f128,
        r_np: f128,
        data_polymers: PolymersData,
        a_cell: f128,
        np_conc: f128,
        cell_conc: f128,
        binding_model: BindingModel,
        polymer_model: PolymerModel,
    ) -> Self {
        let nm = 1.0;
        let nm2 = nm * nm;
        let nm3 = nm * nm * nm;
        let rhostd = 6.023e23 / (1e24 * nm3);

        Self {
            data_polymers,
            kt,
            nm,
            nm2,
            nm3,
            rhostd,
            r_np,
            polymer_model,
            binding_model,
            a_cell,
            np_conc,
            cell_conc,
        }
    }

    /// Calculate the average end-to-end distance in a polymer
    pub fn r_ee(&self, n: f128, a: f128) -> f128 {
        match self.polymer_model {
            PolymerModel::Gaussian | PolymerModel::Flory => n.sqrt() * a,
            PolymerModel::SelfAvoiding => {
                let nu = 3.0 / 5.0;
                n.powf(nu) * a
            }
        }
    }

    /// Calculate area-weighted average bond strength K_LR(h) between ligand-receptor pairs
    /// assuming the planes containing ligands and receptors are parallel and at a distance 'h'
    pub fn k_lr(&self, h: f128, n: f128, a: f128, k_bind_0: f128) -> f128 {
        match self.polymer_model {
            PolymerModel::Gaussian | PolymerModel::Flory => {
                let r_ee2 = self.r_ee(n, a).powi(2);
                let prefactor = k_bind_0 * (12.0 / (PI * r_ee2)).sqrt();

                if h == 0.0 {
                    prefactor / 2.0_f128.sqrt()
                } else {
                    let exp_term = (-3.0 * h * h / (4.0 * r_ee2)).exp();
                    let erf_num = erf(((3.0 * h * h) / (4.0 * r_ee2)).sqrt());
                    let erf_den = erf(((3.0 * h * h) / (2.0 * r_ee2)).sqrt());
                    prefactor * exp_term * erf_num / erf_den
                }
            }
            PolymerModel::SelfAvoiding => {
                panic!("Area-weighted bond cost for SelfAvoiding polymer model not implemented")
            }
        }
    }

    /// Solve for probabilities p_L and p_R of Ligand/Receptor being unbound
    pub fn unbinding_probs(&self, sigma_l: f128, sigma_r: f128, k_lr: f128) -> (f128, f128) {
        let term1 = 4.0 * sigma_l * k_lr;
        let term2 = (1.0 + (sigma_r - sigma_l) * k_lr).powi(2);
        let p_l =
            ((sigma_l - sigma_r) * k_lr - 1.0 + (term1 + term2).sqrt()) / (2.0 * sigma_l * k_lr);

        let term3 = 4.0 * sigma_r * k_lr;
        let term4 = (1.0 + (sigma_l - sigma_r) * k_lr).powi(2);
        let p_r =
            ((sigma_r - sigma_l) * k_lr - 1.0 + (term3 + term4).sqrt()) / (2.0 * sigma_r * k_lr);

        (p_l, p_r)
    }

    /// Calculate steric repulsion free energy per unit area for a specific polymer
    pub fn w_polymer(&self, h: f128, data: &PolymerData, verbose: bool) -> f128 {
        assert!(
            self.polymer_model == PolymerModel::Gaussian
                || self.polymer_model == PolymerModel::Flory,
            "Repulsion implemented only for gaussian/Flory polymer"
        );

        if h == 0.0 {
            f128::INFINITY
        } else {
            let n = data.n;
            let a = data.a;
            let sigma = data.sigma;
            let result = -self.kt * sigma * erf(((3.0 * h * h) / (2.0 * n * a * a)).sqrt()).ln();

            if verbose {
                println!("Steric repulsion from polymer {}: {}", data.name, result as f64);
            }
            result
        }
    }

    /// Calculate bonding contribution to free energy per unit area
    pub fn w_bond(&self, h: f128, sigma_r: f128, k_bind_0: f128, verbose: bool) -> f128 {
        let a = self.data_polymers.ligands.a;
        let n = self.data_polymers.ligands.n;
        let sigma_l = self.data_polymers.ligands.sigma;

        let k = self.k_lr(h, n, a, k_bind_0);
        let (p_l, p_r) = self.unbinding_probs(sigma_l, sigma_r, k);

        // Free energy calculation per unit area
        let w1 = if p_l == 0.0 { f128::INFINITY } else { sigma_l * (p_l.ln() + 0.5 * (1.0 - p_l)) };

        let w2 = if p_r == 0.0 { f128::INFINITY } else { sigma_r * (p_r.ln() + 0.5 * (1.0 - p_r)) };

        let result = (w1 + w2) * self.kt;

        if verbose {
            println!("Bond energy density: {}", result as f64);
        }

        result
    }

    /// Calculate steric repulsion free energy per unit area (all polymers)
    pub fn w_steric(&self, h: f128, verbose: bool) -> f128 {
        let mut result = 0.0;
        result += self.w_polymer(h, &self.data_polymers.short, verbose);
        result += self.w_polymer(h, &self.data_polymers.ligands, verbose);
        result
    }

    /// Calculate total interaction free energy per unit area
    pub fn w_total(&self, h: f128, sigma_r: f128, k_bind_0: f128, verbose: bool) -> f128 {
        let n_long = self.data_polymers.ligands.n;
        let a = self.data_polymers.ligands.a;
        let r_ee = self.r_ee(n_long, a);

        let w_bond = self.w_bond(h, sigma_r, k_bind_0, verbose);
        let w_steric = self.w_steric(h, verbose);

        if verbose {
            println!(
                "h/Ree {:.6} W_bond: {:.6}, W_steric: {:.6}",
                (h / r_ee) as f64,
                w_bond as f64,
                w_steric as f64
            );
        }

        w_bond + w_steric
    }

    /// Calculate binding constant using Derjaguin approximation
    pub fn calculate_binding_constant(
        &self,
        k_bind_0: f128,
        sigma_r: f128,
        z_max: Option<f128>,
        verbose: bool,
    ) -> f128 {
        let force = |h: f128| -> f128 {
            let w_total = self.w_total(h, sigma_r, k_bind_0, verbose);
            if verbose {
                println!("W_total: {} (kbT/nm^2)", w_total as f64);
                println!("Force: {} (kbT/nm)", (2.0 * PI * self.r_np * w_total) as f64);
            }
            2.0 * PI * self.r_np * w_total
        };
        match self.binding_model {
            BindingModel::Saddle => {
                self.calculate_binding_constant_saddle(k_bind_0, sigma_r, z_max, verbose, force)
            }
            BindingModel::Exact => self.calculate_binding_constant_exact(z_max, verbose, force),
        }
    }

    fn calculate_binding_constant_saddle<F>(
        &self,
        k_bind_0: f128,
        sigma_r: f128,
        z_max: Option<f128>,
        verbose: bool,
        force: F,
    ) -> f128
    where
        F: Fn(f128) -> f128,
    {
        let n_long = self.data_polymers.ligands.n;
        let a = self.data_polymers.ligands.a;

        let z_max = z_max.unwrap_or(n_long * a);

        // Find equilibrium binding distance (simplified - using grid search)
        let r_ee = self.r_ee(n_long, a);
        let mut z_bind = r_ee;
        let mut min_w = f128::INFINITY;

        // Simple grid search for minimum
        let n_points = 100;
        for i in 0..n_points {
            let h = (i as f128 / n_points as f128) * z_max;
            if h > 0.0 {
                let w = self.w_total(h, sigma_r, k_bind_0, false);
                if w < min_w {
                    min_w = w;
                    z_bind = h;
                }
            }
        }

        if z_bind > z_max {
            panic!("Equilibrium binding distance is too large");
        }

        if verbose {
            println!("Equilibrium binding distance, normalized to Ree: {}", (z_bind / r_ee) as f64);
            println!(
                "Equilibrium binding distance, normalized to max linear extension: {}",
                (z_bind / z_max) as f64
            );
        }

        // Calculate second derivative at minimum (numerical)
        let dh = 1e-8;
        let f_prime = -(force(z_bind + dh) - force(z_bind - dh)) / (2.0 * dh);

        if f_prime < 0.0 {
            return f128::INFINITY;
        }

        // Binding constant using saddle point approximation
        let area = PI * n_long * a * a;

        // Numerical integration from z_bind to z_max
        let n_steps = 100;
        let mut energy_min = 0.0;
        let step = (z_max - z_bind) / n_steps as f128;
        for i in 0..n_steps {
            let h = z_bind + (i as f128 + 0.5) * step;
            energy_min += force(h) * step;
        }

        if verbose {
            println!("Energy minimum: {}", energy_min as f64);
            println!("Second derivative at minimum: {}", f_prime as f64);
        }

        area * (-energy_min / self.kt).exp() * (2.0 * PI / (f_prime / self.kt)).sqrt()
    }

    fn calculate_binding_constant_exact<F>(
        &self,
        z_max: Option<f128>,
        verbose: bool,
        force: F,
    ) -> f128
    where
        F: Fn(f128) -> f128,
    {
        let n_long = self.data_polymers.ligands.n;
        let a = self.data_polymers.ligands.a;

        let z_max = z_max.unwrap_or(n_long * a);

        // Numerical integration
        let n_steps = 100;
        let integrand = |h: f128| -> f128 {
            let mut a_h = 0.0;
            let step = (z_max - h) / n_steps as f128;
            for i in 0..n_steps {
                let x = h + (i as f128 + 0.5) * step;

                a_h += force(x) * step;
            }
            if verbose {
                println!("h {}, A(h) {}", h as f64, a_h as f64);
            }
            (-a_h / self.kt).exp()
        };

        let a_l = PI * n_long * a * a;

        let mut k_bind = 0.0;
        let step = z_max / n_steps as f128;
        for i in 0..n_steps {
            let h = (i as f128 + 0.5) * step;
            k_bind += integrand(h) * step;
        }
        k_bind *= a_l;

        k_bind
    }

    /// Calculate fraction of bound nanoparticles
    pub fn calculate_bound_fraction(&self, k_bind: f128, verbose: bool) -> f128 {
        if k_bind == f128::INFINITY {
            return 1.0;
        }

        let m_conc = (self.a_cell / (PI * self.r_np * self.r_np)) * self.cell_conc;

        let term = (self.np_conc + m_conc) * k_bind + 1.0;
        let sqrt_term = (term * term - 4.0 * self.np_conc * m_conc * k_bind * k_bind).sqrt();

        let np_m_conc = (term - sqrt_term) / (2.0 * k_bind);

        let bound_fraction = np_m_conc / self.np_conc;

        if verbose {
            println!(
                "term: {}, sqrt_term: {}, NP_M_conc: {}",
                term as f64, sqrt_term as f64, np_m_conc as f64
            );
            println!(
                "M concentration / NP_conc: {}, bound fraction: {}",
                (m_conc / self.np_conc) as f64,
                bound_fraction as f64
            );
        }

        bound_fraction
    }
}

/// Error function approximation (since std lib doesn't have it)
/// Using Abramowitz and Stegun approximation
fn erf(x: f128) -> f128 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

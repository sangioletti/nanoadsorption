use std::f128::consts::PI;

// ============================================================================
// UNITS (from units.py)
// ============================================================================
// Base units
pub const NM: f128 = 1.0; // 1 nm in units of length
pub const NM2: f128 = NM * NM; // 1 nm^2 in units of area
pub const NM3: f128 = NM * NM * NM; // 1 nm^3 in units of volume
pub const MM: f128 = 1e6 * NM; // 1 mm in units of length
pub const MM3: f128 = MM * MM * MM; // 1 mm^3 in units of volume
pub const UM: f128 = 1.0e3 * NM; // 1 micrometer
pub const UM2: f128 = UM * UM; // 1 um^2 in units of area
pub const KT: f128 = 1.0; // 1 kT in units of energy
pub const G: f128 = 1.0; // 1 g in units of mass (bookkeeping only)

// Derived units
pub const L: f128 = 1e24 * NM3; // 1 L in units of volume
pub const ML: f128 = 1e-3 * L; // 1 mL in units of volume
pub const M: f128 = 6.023e23 / L; // Avogadro's number (molar concentration)
pub const NM_CONC: f128 = M * 1e-9; // 1 nM

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate the number of monomers in a PEG chain of a given molecular weight
pub fn nmonomers(mw: f128) -> f128 {
    (mw - 18.0) / 44.0
}

// ============================================================================
// POLYMER DATA STRUCTURE
// ============================================================================

#[derive(Debug, Clone)]
pub struct PolymerData {
    pub n: f128,      // Number of monomers
    pub a: f128,      // Monomer size
    pub sigma: f128,  // Surface density
    pub name: String, // Name identifier
}

// ============================================================================
// SYSTEM PARAMETERS (from system_variables.py)
// ============================================================================
#[derive(Debug, Clone)]
pub struct SystemParameters {
    // Cell and target parameters
    pub n_lympho: f128,
    pub n_t_cells: f128,
    pub v_spleen: f128,
    pub cell_conc: f128,
    pub a_cell: f128,

    // Dosing parameters
    pub npdosing: f128,
    pub vdosing: f128,
    pub f_tzone: f128,
    pub v_tzone: f128,
    pub np_conc: f128,

    // Nanoparticle design
    pub r_np: f128,
    pub n_ligands: i32,
    pub sigma_l: f128,
    pub sigma_p2k: f128,

    // PEG chain properties
    pub amono: f128,
    pub nmono_ligands: f128,
    pub nmono_short: f128,
    pub peg2k_ee: f128,
    pub peg_ligands_ee: f128,
    pub peg_ligands_max_extension: f128,

    // Binding volume
    pub v_bind: f128,

    // Binding constants
    pub kd: f128,
    pub k_bind_0: f128,

    // Polymer data
    pub data_polymers: PolymersData,

    pub verbose: bool,
}

#[derive(Debug, Clone)]
pub struct PolymersData {
    pub short: PolymerData,
    pub ligands: PolymerData,
}

impl Default for SystemParameters {
    fn default() -> Self {
        // Define parameters describing the target: T cells in the spleen
        let n_lympho = 7.5e7; // Average number of lymphocytes in mouse spleen
        let n_t_cells = 0.25 * n_lympho; // Number of T cells in mouse spleen
        let v_spleen = 100.0 * MM3; // Volume of mouse spleen
        let cell_conc = n_t_cells / v_spleen; // T cell concentration in the spleen
        let a_cell = 100.0 * UM2; // Cell area

        // Dosing particles intravenously to animal
        let npdosing = 8e12 / ML; // Number of particles in dosing solution per mL [mL-1]
        let vdosing = 5.0 * 0.02 * ML; // Dosing volume for each animal [mL]
        let f_tzone = 0.1; // Fraction of dosed particles that ends up in T zone
        let v_tzone = 0.5 * 0.084 * ML; // Volume of spleen Tzone in animal [mL]
        let np_conc = npdosing * vdosing * f_tzone / v_tzone; // Particles per mL in Tzone

        // Nanoparticle design
        let r_np = 35.0 * NM; // Nanoparticle radius
        let n_ligands = 150; // Number of ligands on the nanoparticle
        let sigma_l = 150.0 / (4.0 * PI * r_np * r_np); // Surface density of ligands
        let sigma_p2k = 1.0 / (2.0 * NM).powi(2); // Surface density of short PEG chains

        // PEG chain properties
        let amono = 0.28 * NM; // Monomer size in PEG chain
        let nmono_ligands = nmonomers(3400.0 * G); // Number of monomers (3.4K PEG)
        let nmono_short = nmonomers(2000.0 * G); // Number of monomers (2K PEG)
        let peg2k_ee = nmono_short.sqrt() * amono; // End-to-end distance of short PEG
        let peg_ligands_ee = nmono_ligands.sqrt() * amono; // End-to-end of ligand PEG
        let peg_ligands_max_extension = nmono_ligands * amono; // Max extension of ligand PEG

        // Calculate binding volume
        let a1 = PI / 3.0;
        let a2 = ((r_np + peg_ligands_max_extension).powi(2) - (r_np + peg2k_ee).powi(2))
            * (r_np + peg2k_ee);
        let a3 =
            -2.0 * r_np.powi(3) * (1.0 - (r_np + peg2k_ee) / (r_np + peg_ligands_max_extension));
        let v_bind = a1 * (a2 + a3);

        // Binding constant for ligand-receptor binding in solution
        let kd = 1.0 * NM_CONC; // Dissociation constant
        let k_bind_0 = kd.recip(); // Binding constant

        // Create polymer data structures
        let short_polymer =
            PolymerData { n: nmono_short, a: amono, sigma: sigma_p2k, name: "PEG2K".to_string() };

        let ligands_polymer =
            PolymerData { n: nmono_ligands, a: amono, sigma: sigma_l, name: "ligands".to_string() };

        let data_polymers = PolymersData { short: short_polymer, ligands: ligands_polymer };

        SystemParameters {
            n_lympho,
            n_t_cells,
            v_spleen,
            cell_conc,
            a_cell,
            npdosing,
            vdosing,
            f_tzone,
            v_tzone,
            np_conc,
            r_np,
            n_ligands,
            sigma_l,
            sigma_p2k,
            amono,
            nmono_ligands,
            nmono_short,
            peg2k_ee,
            peg_ligands_ee,
            peg_ligands_max_extension,
            v_bind,
            kd,
            k_bind_0,
            data_polymers,
            verbose: false,
        }
    }
}

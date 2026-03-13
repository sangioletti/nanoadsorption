import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.integrate import quad, cumulative_trapezoid
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.special import gamma as scipy_gamma
from mpmath import mp
import math

class MultivalentBinding:
    def __init__(self, kT, R_NP, data_polymers,
                 A_cell, NP_conc, cell_conc,
                 binding_model = None,
                 polymer_model = "gaussian",
                 nonspec_interaction = 0.0,
                 single_receptor = True):
        self.data_polymers = data_polymers
        self.kT = kT # Thermal energy (kB*T)
        self.nm = 1.0 # Sets the units of length
        self.nm2 = ( self.nm )**2 # Sets the units of area
        self.nm3 = ( self.nm )**3 # Sets the units of volume
        self.rhostd = 6.023e23/ ( 1e24 * self.nm3 )
        self.R_NP = R_NP
        self.nonspec_K = np.exp( self.kT * nonspec_interaction ) # 0.0
        self.polymer_model = polymer_model
        self.binding_model = binding_model
        self.A_cell = A_cell
        self.NP_conc = NP_conc
        self.cell_conc = cell_conc
        self.single_receptor = single_receptor

        # Precompute constant derived quantities
        self.NP_excluded_area =  ( 2 * R_NP )**2

        # Identify ligand types: polymers with K_bind_0 > 0
        self._ligand_keys = [k for k, p in data_polymers.items()
                             if p.get("K_bind_0", 0) > 0]

        # Validate single_receptor consistency
        if single_receptor:
            for k in self._ligand_keys:
                if "sigma_R" in data_polymers[k]:
                    raise ValueError(
                        f"single_receptor=True but ligand '{k}' specifies its own sigma_R. "
                        "Either set single_receptor=False or remove sigma_R from ligand entries.")
        else:
            for k in self._ligand_keys:
                if "sigma_R" not in data_polymers[k]:
                    raise ValueError(
                        f"single_receptor=False but ligand '{k}' lacks sigma_R. "
                        "Each ligand must specify its cognate receptor density.")

        # Precompute K_LR parameters for each ligand type
        self._ligand_klr_params = {}
        self._saw_g_interp = {}
        max_ligand_length = 0.0
        max_Ree = 0.0

        for key in self._ligand_keys:
            poly = data_polymers[key]
            N_lig = poly["N"]
            a_lig = poly["a"]
            a_kuhn = poly.get("akuhn", a_lig)
            R_ee = self.r_ee(N_lig, a_lig, a_kuhn)
            max_ligand_length = max(max_ligand_length, N_lig * a_lig)
            if R_ee > max_Ree:
                max_Ree = R_ee

            if self.polymer_model in ("gaussian", "Flory-approx"):
                r_ee2 = R_ee**2
                self._ligand_klr_params[key] = {
                    "prefactor_factor": np.sqrt(6.0 / (np.pi * r_ee2)),
                    "exp_factor": 3.0 / (2.0 * r_ee2),
                    "steric_factor": 3.0 / (2.0 * r_ee2),
                }
            elif self.polymer_model == "Flory-exact":
                z_lig, pz_lig, g_lig = self._compute_saw_pz_g(R_ee)
                self._ligand_klr_params[key] = {
                    "pz_interp": interp1d(
                        z_lig, pz_lig, kind="linear",
                        bounds_error=False, fill_value=0.0,
                    ),
                }
                self._saw_g_interp[poly["name"]] = interp1d(
                    z_lig, g_lig, kind="linear",
                    bounds_error=False, fill_value=(0.0, 1.0),
                )

        self._max_ligand_length = max_ligand_length
        self._Ree_ligands = max_Ree

        # Precompute steric factors for each polymer type (keyed by "name" field)
        self._polymer_steric_factor = {}
        self._polymer_max_length = {}
        self._polymer_cos_theta_max = {}
        self._fjc_g_interp = {}
        self._brush_H = {}
        for key, poly in data_polymers.items():
            N_p = poly["N"]
            a_p = poly["a"]
            self._polymer_max_length[poly["name"]] = N_p * a_p
            if self.polymer_model in ("gaussian", "Flory-approx"):
                a_kuhn_p = poly.get("akuhn", a_p)
                R_ee_p = self.r_ee(N_p, a_p, a_kuhn_p)
                self._polymer_steric_factor[poly["name"]] = 3.0 / (2.0 * R_ee_p**2)
            elif self.polymer_model == "Flory-exact":
                a_kuhn_p = poly.get("akuhn", a_p)
                R_ee_p = self.r_ee(N_p, a_p, a_kuhn_p)
                # Only compute if not already done in ligand loop
                if poly["name"] not in self._saw_g_interp:
                    z_p, pz_p, g_p = self._compute_saw_pz_g(R_ee_p)
                    self._saw_g_interp[poly["name"]] = interp1d(
                        z_p, g_p, kind="linear",
                        bounds_error=False, fill_value=(0.0, 1.0),
                    )

    @staticmethod
    def _compute_saw_pz_g(R_ee, n_pts=400, R_max_factor=4.0):
        """Half-space z-marginal P_z(z) and CDF g(h) for a 3D SAW using the
        des Cloizeaux form for the end-to-end radial distribution.

        P(R) = C * (R/R_ee)^theta * exp(-D*(R/R_ee)^delta)
        with 3D SAW exponents: nu ≈ 0.588, gamma ≈ 1.157,
        theta = (gamma-1)/nu, delta = 1/(1-nu).
        Normalized so int P(R)dR = 1 and int R^2 P(R)dR = R_ee^2.

        Half-space P_z(z) = int_z^infty P(R)/R dR, normalized so
        int_0^infty P_z dz = 1.
        g(h) = int_0^h P_z(z) dz.

        Uses scipy.integrate.quad for each z point to correctly handle
        the integrable singularity P(R)/R ~ R^(theta-1) near R=0.

        Returns (z_pts, pz_pts, g_pts) on z in [0, R_max].
        """
        nu = 0.588   # 3D SAW
        gamma = 1.157
        theta = (gamma - 1.0) / nu
        delta = 1.0 / (1.0 - nu)
        # D from moment condition: <R^2> = R_ee^2
        D = (scipy_gamma((theta + 3.0) / delta) / scipy_gamma((theta + 1.0) / delta)) ** (delta / 2.0)
        # C from int P dR = 1: C * R_ee * (1/delta)*D^(-(theta+1)/delta)*Gamma((theta+1)/delta) = 1
        I0 = (1.0 / delta) * D ** (-(theta + 1.0) / delta) * scipy_gamma((theta + 1.0) / delta)
        C = 1.0 / (R_ee * I0)

        R_max = R_max_factor * R_ee

        def _P_over_R(R):
            x = R / R_ee
            return C * (x ** theta) * np.exp(-D * (x ** delta)) / R

        # Evaluate P_z(z) = int_z^R_max P(R)/R dR using adaptive quadrature
        z_pts = np.linspace(0, R_max, n_pts + 1)[1:]  # exclude z=0
        z_pts = np.concatenate([[1e-4 * R_ee], z_pts[z_pts > 1e-4 * R_ee]])
        P_z_raw = np.array([quad(_P_over_R, z, R_max * 2)[0] for z in z_pts])
        P_z_raw = np.maximum(P_z_raw, 0.0)

        # Normalize so int P_z dz = 1 (should already be ~1 by construction)
        norm = np.trapezoid(P_z_raw, z_pts)
        if norm <= 0:
            norm = 1.0
        P_z_pts = P_z_raw / norm
        g_pts = cumulative_trapezoid(P_z_pts, z_pts, initial=0.0)
        g_pts = np.minimum(g_pts, 1.0)
        return z_pts, P_z_pts, g_pts

    def _K_LR_single(self, h, ligand_key):
        """Area-weighted bond strength K_LR(h) for a single ligand type.

        Uses K_bind_0 from data_polymers[ligand_key]["K_bind_0"] and the
        precomputed K_LR parameters for that ligand.

        Gaussian/Flory: K_bind_0 * P_z(h) / erf(h*sqrt(3/(2Na^2))).
        Flory-exact: K_bind_0 * P_z(h) / g(h)  from des Cloizeaux SAW."""
        K_bind_0 = self.data_polymers[ligand_key]["K_bind_0"]
        params = self._ligand_klr_params[ligand_key]

        if self.polymer_model in [ "gaussian", "Flory-approx" ]:
            g_lig = math.erf(h * math.sqrt(params["steric_factor"]))
            if g_lig < 1e-15:
                g_lig = 1e-15
            return K_bind_0 * params["prefactor_factor"] * np.exp(-h**2 * params["exp_factor"]) / g_lig
        elif self.polymer_model == "Flory-exact":
            pz = float(params["pz_interp"](h))
            lig_name = self.data_polymers[ligand_key]["name"]
            g_lig = float(self._saw_g_interp[lig_name](h))
            if g_lig < 1e-15:
                g_lig = 1e-15
            return K_bind_0 * pz / g_lig
        else:
            raise NotImplementedError( f'Area-weighted bond cost for polymer model {self.polymer_model} not implemented' )    
        
    def r_ee( self, N, a, a_kuhn ):
        """Calculate the average end-to-end distance in a polymer.
        For a rigid rod this is simply the rod length L = N*a."""
        if self.polymer_model in ("gaussian"):
            #R_ee = np.sqrt( N ) * a
            R_ee =  np.sqrt( N * a * a_kuhn)
        elif self.polymer_model in ( "Flory-approx", "Flory-exact" ):
            nu = 3.0/5.0
            R_ee = ( a * N )**nu * a_kuhn**(1-nu)
        else:
            raise ValueError( f'Polymer model {self.polymer_model} not recognized' )
        return R_ee
    
    def unbinding_probs( self, sigma_L, sigma_R, K_LR ):
        """Solve for probabilities p_L and p_R of Ligand/Receptor being UNbound"""
        cond1 = self.binding_model == "exact" 
        cond2 = self.binding_model == "saddle"
        error_string = f"""Only usable with exact and saddle point models, detected {self.binding_model}"""
        assert ( cond1 or cond2 ), AssertionError( error_string )
        sigma_L = mp.mpf(sigma_L)
        sigma_R = mp.mpf(sigma_R)
        check = isinstance( K_LR, np.ndarray)
        
        if check:
            K_LR = mp.mpf(K_LR[0])
        else:
            K_LR = mp.mpf(K_LR)

        p_L = (sigma_L-sigma_R) * K_LR - 1.0 + mp.sqrt(4.0*sigma_L*K_LR + (1.0+(sigma_R-sigma_L)*K_LR)**2)
        p_L /= (2.0*sigma_L*K_LR)
        
        p_R = (sigma_R-sigma_L) * K_LR - 1.0 + mp.sqrt(4.0*sigma_R*K_LR + (1.0+(sigma_L-sigma_R)*K_LR)**2)
        p_R /= (2.0*sigma_R*K_LR)
        
        return p_L, p_R
    
    def W_bond(self, h, sigma_R=None, verbose = False ):
        """Calculate bonding contribution to free energy per unit area.

        Sums over all ligand types (independent, non-competing pairs).
        Each ligand type uses its own K_bind_0, sigma_L, and sigma_R.

        Parameters:
            h: surface separation
            sigma_R: receptor surface density. Used for all ligands when
                single_receptor=True. Ignored when single_receptor=False
                (each ligand reads sigma_R from data_polymers).
            verbose: print diagnostic output
        """
        res = 0.0
        for key in self._ligand_keys:
            poly = self.data_polymers[key]
            sigma_L_i = poly["sigma"]

            if self.single_receptor:
                sigma_R_i = sigma_R
            else:
                sigma_R_i = poly["sigma_R"]

            if sigma_R_i is None or sigma_R_i == 0:
                continue

            K_i = self._K_LR_single(h, key)
            if K_i == 0.0 or K_i * max(sigma_L_i, sigma_R_i) < 1e-12:
                continue

            p_L, p_R = self.unbinding_probs(sigma_L_i, sigma_R_i, K_i)

            if p_L == 0:
                W1 = np.inf
            else:
                W1 = sigma_L_i * (mp.log(p_L) + 0.5*(1 - p_L))

            if p_R == 0:
                W2 = np.inf
            else:
                W2 = sigma_R_i * (mp.log(p_R) + 0.5*(1 - p_R))

            W_bond_i = ( W1 + W2 ) * self.kT
            if verbose:
                print( f'Bond energy density from ligand {key}: {W_bond_i}' )
            res += W_bond_i

        return res
    
    def W_steric(self, h, verbose = False):
        """Calculate steric repulsion free energy per unit area"""
        assert self.polymer_model in [ "gaussian", "Flory-approx", "Flory-exact" ], \
            AssertionError( f"Repulsion not implemented for polymer model {self.polymer_model}")

        res = 0.0
        for polymer in self.data_polymers.values():
            res += self.W_polymer( h, polymer, verbose = verbose )
        return res
    
    def W_polymer( self, h, data, verbose = False):
        """Calculate steric repulsion free energy per unit area.

        Gaussian/Flory: -kT * sigma * ln(erf(h * sqrt(3/(2*N*a^2))))
        Flory-exact: -kT * sigma * ln(g(h)) where g is the SAW CDF from des Cloizeaux."""
        assert self.polymer_model in [ "gaussian", "Flory-approx", "Flory-exact" ], \
            AssertionError( f"Repulsion not implemented for polymer model {self.polymer_model}")

        if isinstance( h, np.ndarray):
            h = mp.mpf(h[0])
        else:
            h = mp.mpf(h)

        if h == 0:
            res = np.inf
        elif self.polymer_model in ("gaussian", "Flory-approx"):
            name = data["name"]
            sigma = data[ "sigma" ]
            steric_factor = self._polymer_steric_factor[name]
            res = -self.kT * sigma * mp.log(mp.erf( mp.sqrt( steric_factor * h**2 )))
        elif self.polymer_model == "Flory-exact":
            name = data["name"]
            sigma = data["sigma"]
            g = float(self._saw_g_interp[name](float(h)))
            if g <= 0:
                res = np.inf
            else:
                res = -self.kT * sigma * mp.log(mp.mpf(g))

        if verbose:
            print( f'Steric repulsion from polymer {data["name"]}: {res}' )
        return res
    
    def W_total(self, h, sigma_R=None, verbose = False):
        """Calculate total interaction free energy per unit area"""
        W_bond = self.W_bond(h, sigma_R, verbose)
        W_steric = self.W_steric( h, verbose)
        if verbose:
            Ree = self._Ree_ligands
            print( f'h/Ree {h/Ree} W_bond: {W_bond}, W_steric: {W_steric}' )
        return W_bond + W_steric
    
    def calculate_binding_constant( self, sigma_R=None,
                                   z_max = None,
                                   verbose = False,
                                    ):

            if not self._ligand_keys:
                return 0.0

            R_NP = self.R_NP
            Area = self.NP_excluded_area  # Approximate adsorption area of the nanoparticle
            z_max = self._max_ligand_length

            if self.binding_model == "saddle":
                def force(h):
                    return 2 * np.pi * R_NP * self.W_total(h,
                                                       sigma_R = sigma_R,
                                                       verbose = verbose
                                                       )

                # Find equilibrium binding distance where W_total = 0
                def find_z_bind(h):
                    return self.W_total(h,
                                    sigma_R = sigma_R,
                                    verbose = verbose
                                    )

                R_ee = self._Ree_ligands

                initial_guess = R_ee
                bounds = [(0.0, z_max)]
                result = minimize(find_z_bind, initial_guess, bounds=bounds)
                z_bind = result.x[ 0 ]

                if z_bind > z_max:
                    raise ValueError( f'Equilibrium binding distance is too large z/R_ee: {z_bind/R_ee}' )

                if verbose:
                    print( f'Equilibrium binding distance, normalized to Ree: {z_bind / R_ee }' )
                    print( f'Equilibrium binding distance, normalized to max linear extension: {z_bind / z_max }' )

                # Calculate the second derivative of A at minimum. This is minus the first derivative of the force
                dh = 1e-8  # Small step for numerical derivative
                F_prime = -(force(z_bind + dh) - force(z_bind - dh))/(2*dh)
                try:
                    assert F_prime >= 0.0, AssertionError( f'Second derivative at minimum: {F_prime} should not be negative' )
                except:
                    K_bind = np.inf
                    return K_bind

                # Binding constant using saddle point approximation
                h_grid_saddle = np.linspace(z_bind, z_max, 100)
                energy_min = np.trapz([force(h) for h in h_grid_saddle], h_grid_saddle)

                if verbose:
                    print( f'Energy minimum: {energy_min}' )
                    print( f'Second derivative at minimum: {F_prime}' )

                K_bind = Area * mp.exp(-energy_min/self.kT) * mp.sqrt(2*mp.pi/(F_prime/self.    kT))
                return K_bind

            elif self.binding_model == "exact":
                # Precompute force on a single grid (instead of 100x100 redundant evaluations)
                n_grid = 200
                h_grid = np.linspace(0, z_max, n_grid)
                force_grid = np.array([
                    float(2 * mp.pi * R_NP * self.W_total(h, sigma_R=sigma_R,
                                                          verbose=verbose))
                    for h in h_grid
                ])

                # A(h) = integral from h to z_max of force(x)dx
                # Compute via reverse cumulative trapezoid for all grid points at once
                A_grid = np.zeros(n_grid)
                # cumulative_trapezoid on reversed arrays gives cumulative integral from z_max backwards
                A_grid[:-1] = cumulative_trapezoid(force_grid[::-1], h_grid[::-1])[::-1] * (-1)
                # A_grid[-1] = 0 (integral from z_max to z_max)

                integrand_grid = np.array([float(mp.exp(-A_grid[i] / self.kT))
                                           for i in range(n_grid)])
                K_bind = Area * np.trapezoid(integrand_grid, h_grid)

                return K_bind * self.nonspec_K

    def calculate_K_bind_vs_receptors(self, max_N_receptor, verbose=False):
        """Precompute K_bind for integer receptor numbers.

        single_receptor=True: returns 1D array K_bind[NR], NR = 0..max_N_receptor-1.
            All ligands see sigma_R = NR / A_excl.

        single_receptor=False: returns (K_bind_flat, grid_shape, NR_aves) where
            K_bind_flat is a 1D array over the flattened Cartesian product of
            receptor numbers for each ligand type.
            grid_shape is the tuple of axis lengths.
            NR_aves is a list of average receptor numbers per ligand.
        """
        NP_excluded_area = self.NP_excluded_area

        if self.single_receptor:
            K_bind_vs_NR = np.empty(max_N_receptor, dtype=object)
            K_bind_vs_NR[0] = 0.0
            for NR in range(1, max_N_receptor):
                sigma_R_i = NR / NP_excluded_area
                K_bind_vs_NR[NR] = self.calculate_binding_constant(
                    sigma_R=sigma_R_i, verbose=verbose)
            return K_bind_vs_NR

        else:
            # Multi-receptor: build grid over independent receptor types
            import itertools
            n_lig = len(self._ligand_keys)
            # Determine max NR for each receptor type
            axes = []
            NR_aves = []
            for key in self._ligand_keys:
                sigma_R_k = self.data_polymers[key]["sigma_R"]
                NR_ave_k = NP_excluded_area * sigma_R_k
                NR_aves.append(NR_ave_k)
                max_NR_k = min(max_N_receptor,
                               int(NR_ave_k) + 4 * (int(NR_ave_k) + 1) + 1)
                axes.append(range(max_NR_k))

            grid_shape = tuple(len(ax) for ax in axes)
            total_points = 1
            for s in grid_shape:
                total_points *= s

            K_bind_flat = np.empty(total_points, dtype=object)
            for idx, combo in enumerate(itertools.product(*axes)):
                # combo = (NR_1, NR_2, ...) for each ligand type
                # Build sigma_R for each ligand from its NR
                if all(nr == 0 for nr in combo):
                    K_bind_flat[idx] = 0.0
                    continue
                # Temporarily set sigma_R in data_polymers for this evaluation
                saved = {}
                for i, key in enumerate(self._ligand_keys):
                    saved[key] = self.data_polymers[key].get("sigma_R")
                    self.data_polymers[key]["sigma_R"] = combo[i] / NP_excluded_area
                K_bind_flat[idx] = self.calculate_binding_constant(verbose=verbose)
                # Restore
                for key in self._ligand_keys:
                    if saved[key] is not None:
                        self.data_polymers[key]["sigma_R"] = saved[key]

            return K_bind_flat, grid_shape, NR_aves

    def calculate_bound_vs_receptors_monodisperse(self,
                                max_N_receptor,
                                depletion:bool = True,
                                verbose:bool = False,
                                K_bind_vs_NR=None,
                                NP_conc=None):
        """Calculate fraction of nanoparticles bound for each integer NR.

        Only supported for single_receptor=True.
        If K_bind_vs_NR is provided, use pre-computed K_bind values instead of
        recomputing them. If NP_conc is provided, override self.NP_conc."""
        A_cell = self.A_cell
        NP_conc = NP_conc if NP_conc is not None else self.NP_conc
        cell_conc = self.cell_conc
        NP_excluded_area = self.NP_excluded_area
        M_conc = (A_cell/NP_excluded_area) * cell_conc
        bound_vs_receptor = np.zeros( max_N_receptor )


        if depletion:
            for NR in range( 1, max_N_receptor ):
                if K_bind_vs_NR is not None:
                    K_bind = K_bind_vs_NR[NR]
                else:
                    sigma_R_i = NR / NP_excluded_area
                    K_bind = self.calculate_binding_constant( sigma_R_i )
                if K_bind == np.inf:
                    bound_vs_receptor[NR] = M_conc / NP_conc
                else:
                    term = (NP_conc + M_conc) * K_bind + 1
                    sqrt_term = mp.sqrt( mp.power( term, 2 ) - 4 * NP_conc * M_conc * K_bind**2 )
                    NP_M_conc = (term - sqrt_term)/(2 * K_bind)
                    bound_vs_receptor[NR] = NP_M_conc / NP_conc


        else: # Langmuir formula, without depletion
            for NR in range( 1, max_N_receptor ):
                if K_bind_vs_NR is not None:
                    K_bind = K_bind_vs_NR[NR]
                else:
                    sigma_R_i = NR / NP_excluded_area
                    K_bind = self.calculate_binding_constant( sigma_R_i )

                if K_bind == np.inf:
                    bound_vs_receptor[NR] = 1.0
                else:
                    bound_vs_receptor[NR] = NP_conc * K_bind / (1 + NP_conc * K_bind )

        return bound_vs_receptor
    
    
    def calculate_bound_fraction(self, sigma_R=None,
                                 fluctuations=False, depletion=True,
                                 bound_vs_receptor=None,
                                 K_bind_vs_receptors=None,
                                 verbose=False, max_factor=4,
                                 max_n_receptor=200,
                                 NP_conc=None, rho_m=None):
        """Calculate the fraction of nanoparticles bound to the cell surface.

        Parameters:
            sigma_R: receptor surface density. Required when single_receptor=True.
                Ignored when single_receptor=False (each ligand has its own sigma_R).
            fluctuations: if True, Poisson-average over receptor number fluctuations
            depletion: if True, account for NP depletion; if False, Langmuir (infinite bulk)
            bound_vs_receptor: precomputed bound fractions per NR
                (required when fluctuations=True, depletion=False, single_receptor=True)
            K_bind_vs_receptors: precomputed K_bind array
                (required when fluctuations=True, depletion=True;
                 computed automatically if not provided)
            verbose: print diagnostic output
            max_factor: controls Poisson truncation (fluctuations=True, depletion=False)
            max_n_receptor: max NR for K_bind precomputation
            NP_conc: override self.NP_conc
            rho_m: total binding site concentration (fluctuations=True, depletion=True)
        """
        NP_conc = NP_conc if NP_conc is not None else self.NP_conc
        NP_excluded_area = self.NP_excluded_area

        if not fluctuations:
            # --- Monodisperse: no Poisson averaging over receptor numbers ---
            K_bind = self.calculate_binding_constant(sigma_R)

            if depletion:
                M_conc = (self.A_cell / NP_excluded_area) * self.cell_conc
                if K_bind == np.inf:
                    return min(1.0, M_conc / NP_conc)
                term = (NP_conc + M_conc) * K_bind + 1
                sqrt_term = mp.sqrt(mp.power(term, 2) - 4 * NP_conc * M_conc * K_bind**2)
                NP_M_conc = (term - sqrt_term) / (2 * K_bind)
                bound_fraction = NP_M_conc / NP_conc
                if verbose:
                    print(f'term: {term}, sqrt_term: {sqrt_term}, NP_M_conc: {NP_M_conc}')
                    print(f'M_conc / NP_conc: {M_conc/NP_conc}, bound fraction: {bound_fraction}')
                return bound_fraction
            else:
                if K_bind == np.inf:
                    return 1.0
                return NP_conc * K_bind / (1 + NP_conc * K_bind)

        else:
            # --- Fluctuations: Poisson-average over receptor numbers ---
            if self.single_receptor:
                NR_ave = NP_excluded_area * sigma_R

                if not depletion:
                    # Langmuir with Poisson fluctuations
                    assert bound_vs_receptor is not None, \
                        "bound_vs_receptor required when fluctuations=True, depletion=False"
                    if verbose:
                        print(f"NR_ave: {NR_ave}, sigma_R: {sigma_R}")

                    int_NR_ave = int(NR_ave)
                    max_NR = int_NR_ave + max_factor * (int_NR_ave + 1) if int_NR_ave > 1 else 20
                    if verbose:
                        print(f"Max NR required: {max_NR}, precomputed: {len(bound_vs_receptor)}")
                    assert len(bound_vs_receptor) >= max_NR

                    poisson = np.zeros(max_NR)
                    exp_neg_avg = mp.exp(-NR_ave)
                    for NR in range(1, max_NR):
                        poisson[NR] = poisson_distribution(NR, NR_ave, exp_neg_avg)

                    return min(1, np.sum(bound_vs_receptor[:len(poisson)] * poisson))

                else:
                    # Self-consistent solve with Poisson fluctuations and depletion
                    if K_bind_vs_receptors is None:
                        K_bind_vs_receptors = self.calculate_K_bind_vs_receptors(
                            max_n_receptor)

                    rho_i, K_i, bound_fraction = self.self_consistent_rho(
                        K_bind_vs_receptors, rho_m, NP_conc, NR_ave)
                    return bound_fraction

            else:
                # Multi-receptor: independent Poisson for each receptor type
                import itertools

                if K_bind_vs_receptors is None:
                    K_bind_vs_receptors = self.calculate_K_bind_vs_receptors(
                        max_n_receptor)

                K_bind_flat, grid_shape, NR_aves = K_bind_vs_receptors

                # Build flattened Poisson weight array (product of independent Poissons)
                n_lig = len(self._ligand_keys)
                axes_poisson = []
                for i in range(n_lig):
                    NR_ave_i = NR_aves[i]
                    n_i = grid_shape[i]
                    exp_neg = mp.exp(-NR_ave_i)
                    poisson_i = np.array([float(poisson_distribution(k, NR_ave_i, exp_neg))
                                          for k in range(n_i)])
                    axes_poisson.append(poisson_i)

                # Compute product of marginals for each grid point
                total_points = len(K_bind_flat)
                poisson_flat = np.ones(total_points)
                for idx, combo in enumerate(itertools.product(
                        *[range(s) for s in grid_shape])):
                    weight = 1.0
                    for i, nr_i in enumerate(combo):
                        weight *= axes_poisson[i][nr_i]
                    poisson_flat[idx] = weight

                if depletion:
                    # Use self_consistent_rho with flattened arrays
                    if rho_m is None:
                        rho_m = (self.A_cell / NP_excluded_area) * self.cell_conc
                    rho_m_flat = poisson_flat * rho_m
                    rho_i, K_i, bound_fraction = self.self_consistent_rho_flat(
                        K_bind_flat, rho_m_flat, NP_conc)
                    return bound_fraction
                else:
                    # Langmuir with multi-receptor Poisson fluctuations
                    result = 0.0
                    for idx in range(total_points):
                        K = float(K_bind_flat[idx])
                        if K == 0:
                            continue
                        if K == np.inf:
                            bf_i = 1.0
                        else:
                            bf_i = NP_conc * K / (1 + NP_conc * K)
                        result += poisson_flat[idx] * bf_i
                    if result > 1.0:
                        raise ValueError(f"result: {result} is greater than 1.0")
                    return min(1.0, result)

    def self_consistent_rho( self, K_i, rho_m_tot, NP_conc, NR_ave, tol=1e-16, max_iter=2000 ):
        n = len( K_i )
        K = np.array( [float(K_i[i]) for i in range(n)] )
        rho_m = np.zeros( n )
        for i in range( n ):
            rho_m[i] = poisson_distribution( i, NR_ave )
        rho_m *= rho_m_tot

        # Reduce to scalar equation in c = NP_conc - sum(rho), the free NP concentration.
        # Each rho[i] = K[i]*rho_m[i]*c / (1 + K[i]*c), so:
        # g(c) = NP_conc - c - sum_i[ K[i]*rho_m[i]*c / (1 + K[i]*c) ] = 0
        Km = K * rho_m  # precompute K[i]*rho_m[i]

        def g_and_gprime(c):
            denom = 1.0 + K * c
            g  = NP_conc - c - np.sum( Km * c / denom )
            gp = -1.0 - np.sum( Km / denom**2 )
            return g, gp

        # Bracket: g(0)=NP_conc>0, g(NP_conc)<0
        c_lo, c_hi = 0.0, float(NP_conc)
        c = 0.5 * c_hi  # initial guess

        for n_cycle in range( max_iter ):
            g_val, gp_val = g_and_gprime( c )

            # Update bracket BEFORE bisection fallback so that
            # the fallback uses the tightened bracket
            if g_val > 0:
                c_lo = c
            else:
                c_hi = c

            # Newton step with bisection safeguard
            c_new = c - g_val / gp_val
            if c_new <= c_lo or c_new >= c_hi:
                c_new = 0.5 * (c_lo + c_hi)  # bisection fallback

            if abs(c_new - c) < tol * float(NP_conc):
                c = c_new
                break
            c = c_new

        # Recover rho[i] from the converged c
        denom = 1.0 + K * c
        rho = Km * c / denom

        # Compute bound fraction directly from c to avoid floating-point
        # precision loss when summing rho[i] for large K[i] values
        bound_fraction = 1.0 - c / float(NP_conc)

        #print( f"Self consistent solve finished, cycle {n_cycle}, free NP frac = {c/NP_conc:.6e}" )
        return (rho, K_i, bound_fraction)

    def self_consistent_rho_flat(self, K_i, rho_m, NP_conc, tol=1e-16, max_iter=2000):
        """Self-consistent solve with pre-built rho_m array (for multi-receptor case).

        Same Newton/bisection solver as self_consistent_rho, but rho_m is
        provided directly (already weighted by Poisson products) rather than
        being computed internally from NR_ave."""
        n = len(K_i)
        K = np.array([float(K_i[i]) for i in range(n)])
        rho_m_arr = np.array([float(rho_m[i]) for i in range(n)])

        Km = K * rho_m_arr

        def g_and_gprime(c):
            denom = 1.0 + K * c
            g  = NP_conc - c - np.sum(Km * c / denom)
            gp = -1.0 - np.sum(Km / denom**2)
            return g, gp

        c_lo, c_hi = 0.0, float(NP_conc)
        c = 0.5 * c_hi

        for n_cycle in range(max_iter):
            g_val, gp_val = g_and_gprime(c)
            if g_val > 0:
                c_lo = c
            else:
                c_hi = c
            c_new = c - g_val / gp_val
            if c_new <= c_lo or c_new >= c_hi:
                c_new = 0.5 * (c_lo + c_hi)
            if abs(c_new - c) < tol * float(NP_conc):
                c = c_new
                break
            c = c_new

        denom = 1.0 + K * c
        rho = Km * c / denom
        bound_fraction = 1.0 - c / float(NP_conc)
        return (rho, K_i, bound_fraction)


def poisson_distribution( k, average_k, exp_neg_avg=None ):
    if exp_neg_avg is None:
        exp_neg_avg = mp.exp(-average_k)
    try:
        result = exp_neg_avg * mp.power( average_k, k ) / mp.factorial( k )
    except ValueError:
        # Use Ramanujan approximation for log(k!)
        log_result = -average_k + k * math.log(average_k) - ramanujan_log_factorial(k)
        result = mp.exp(log_result)
    return result

def ramanujan_log_factorial(n):
    """Ramanujan's approximation for log(n!)"""
    return (n * math.log(n) - n 
            + (1/6) * math.log(8*n**3 + 4*n**2 + n + 1/30) 
            + 0.5 * math.log(math.pi))

# This function simply helps calculating the number of monomers in a PEG chain of a given molecular weight
def Nmonomers( MW ):
    '''Calculate the number of monomers in a PEG chain of a given molecular weight'''
    return int( MW - 18 ) / 44.0

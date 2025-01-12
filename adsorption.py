import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve, minimize
from scipy.integrate import quad    
from mpmath import mp

class MultivalentBinding:
    def __init__(self, kT, R_NP, N_long, N_short, a_mono, 
                 A_cell, NP_conc, cell_conc, 
                 binding_model = None, 
                 polymer_model = "gaussian"):
        self.kT = kT # Thermal energy (kB*T)
        self.nm = 1.0 # Sets the units of length
        self.nm2 = ( self.nm )**2 # Sets the units of area
        self.nm3 = ( self.nm )**3 # Sets the units of volume
        self.rhostd = 6.023e23/ ( 1e24 * self.nm3 )
        self.R_NP = R_NP 
        self.N_long = N_long    
        self.N_short = N_short
        self.a_mono = a_mono
        # These are the values used by Lennart and taken from paper.
        # They are evaluated for polymers in the mushroom (2K) and brush (3.4K) regime
        # However, I notice they are taken at inconsistent values of the grafting distance
        # since the 3.4K is smaller which I am not sure is really correct.
        self.R_max_3p4K = 11.1 * self.nm
        self.R_ee_2K = 3.9 * self.nm #Initial calculations by Lennart used 4.0 but difference is irrelevant
        self.polymer_model = polymer_model 
        self.binding_model = binding_model
        self.A_cell = A_cell
        self.NP_conc = NP_conc
        self.cell_conc = cell_conc
        print( 'Few problems:' )
        print( f'1) The values of R_max_3p4K and R_ee_2K are taken at different grafting distances' )
        print( f'This problem might be solvable using a Komura-Safran model for the bimodal brush' )
        print( f"""2) Brush repulsion only counted for long brush in Gaussian state but could be improved
               - and you know how using KS-blob model. However, this should not really matter IF short brush is
              very hard? Check""" )
        print( f'3) Initial value used for Veff was not correctly calculated by Lennart' )
        print( f'4) Bond energy using Flory model + Gaussian approx around a different average might be possible' )

    def K_LR(self, h, N, a, K0):
        """Calculate area-weighted average bond strength K_LR(h) between ligand-receptor pairs,
        assuming the planes containing ligands and receptors are parallel and at a distance 'h'
        from each other.
        K_LR is calculated assuming receptors are fixed point on a surface and ligands are tethered to 
        a NP by a Gaussian polymer chain"""
        if self.polymer_model == "gaussian":
            prefactor = K0 * np.sqrt( 12.0 / ( np.pi * N * a**2)) 
            exp_term = np.exp(-3.0 * h / (4 * N * a**2))
            erf_num = erf(np.sqrt(3 * h**2/(4 * N * a**2)))
            erf_den = erf(np.sqrt(3 * h**2/(2 * N * a**2)))

            if h == 0:
                return prefactor / np.sqrt(2.0)
            else:
                return prefactor * exp_term * erf_num/erf_den
        else:
            raise NotImplementedError( f'Area-weighted bond cost for polymer model {self.polymer_model} not implemented' )    
        
    def r_ee( self, N, a ):
        """Calculate the average end-to-end distance in a polymer"""
        if self.polymer_model in ( "gaussian", "ideal" ):
            R_ee = np.sqrt( N ) * a
        elif self.polymer_model in ( "self_avoiding_walk", "Flory" ):
            nu = 3.0/5.0
            R_ee = N**(nu) * a
        elif self.polymer_model in ( "Alexander-DeGennes", "brush" ):
            R_ee = self.R_ee_2K
        else:
            raise ValueError( f'Polymer model {self.polymer_model} not recognized' )
        return R_ee
    
    def r_ave_bond(self, z_dist, max_bond_length = None ):
        """Calculate the average bond length between a ligand and a receptor, **assuming** all bonds
        have equal probability, the grafting point of the ligand is at distance z_dist from the
        receptor containing surface and that the maximum bond extension is max_bond_length.
        Calculated as:
        w(x) = A_norm * 2 pi x 
        A = int_0_x_max w(x) dx --> A = 1.0 / [ pi * (max_bond_lengt**2 - z_dist**2) ]
        x_ave = int_0,x_max w(x) x dx
        average_bond_length = sqrt( z_dist**2 + x_ave**2 )
        """
        if max_bond_length is None:
            max_bond_length = 3.0 * z_dist
        assert max_bond_length > z_dist # Sanity check
        x_ave = 2.0 / 3.0 * np.sqrt( max_bond_length**2 - z_dist**2 )
        average_bond_length = np.sqrt( z_dist**2 + x_ave**2 ) 
        return average_bond_length
    
    def x_max_span( self, z_dist, max_bond_length = None ):
        """The radius of the maximum circle spanned by a ligand on the receptor-containing surface.
        Same assumption/geoemtry as above"""
        if max_bond_length is None:
            max_bond_length = 3.0 * z_dist
        assert max_bond_length > z_dist # Sanity check
        x_max = np.sqrt( max_bond_length**2 - z_dist**2 ) 
        return x_max 
    
    def geometric_parameters( self, R_NP, R_long, R_short ):
        #Calculate the 'active area' containing RECEPTORS interacting with the NP
        height_cone = R_NP + R_short
        hypotenus_cone = R_NP + R_long
        r_cone_squared = hypotenus_cone**2 - height_cone**2
        A_R = np.pi * r_cone_squared

        #The 'active area' containing ligands LIGANDS interacting with the NP
        # Step1: calculate the cosine of the aperture angle
        cos_alpha = ( R_NP + R_short ) / ( R_NP + R_long )
        assert cos_alpha >= 0.0, AssertionError( f'cos_alpha: {cos_alpha} < 0.0 but should not in this geometry' )
        sin_alpha = np.sqrt( 1.0 - cos_alpha**2 )
        assert cos_alpha <= 1.0, AssertionError( f'cos_alpha: {cos_alpha} > 1.0' )
        f1 = 1.0 - cos_alpha 
        # Step2: multiply by 2*pi*R to obtain the area of the spherical patch
        A_P = 2.0 * np.pi * R_NP**2 * f1

        # Calculate effective volume where ligand-receptors reside:
        # Veff = V_cone - V_small_cone - V_spherical_patch
        height_cone = R_NP + R_short
        V_cone = np.pi / 3.0 * r_cone_squared * height_cone
        V_spherical_patch = 2.0 / 3.0 * np.pi * R_NP**3 * f1
        V_eff = V_cone - V_spherical_patch

        error_string1 = f"R_NP {R_NP}, R_long {R_long} R_short {R_short} \n"
        error_string2 = f"V_cone {V_cone:3.5e} V_spherical_patch {V_spherical_patch:3.5e}"
        assert V_eff > 0, AssertionError( f'Effective volume is negative: {V_eff} \n' + error_string1 + error_string2 )

        return A_R, A_P, V_eff

    def chi_LR(self, r_bond, N, a, K0 ):
        """Calculate average bond strength between ligand-receptor pairs.
        Similar but not the same to K_LR method above, as this is what matter for discrete 'binders'
        (ligands or receptors)"""
        if self.binding_model in ( "simple", "average" ):
            r_ee = self.r_ee( N, a )
            if self.polymer_model == "gaussian":
                chi_conf = mp.mpf( ( 3.0 / (2.0 * np.pi * r_ee**2))**(3.0/2.0) * np.exp(-3*r_bond**2/(2*r_ee**2)) )
            else:
                raise NotImplementedError( f'Bond cost for polymer model {self.polymer_model} not implemented' )  
        if self.binding_model == "shaw":
            R_NP = self.R_NP
            R_long = self.R_max_3p4K
            R_short = self.R_ee_2K
            _, _, V_eff = self.geometric_parameters( R_NP = R_NP, R_long = R_long , R_short = R_short )
            chi_conf = K0 / V_eff
        return K0 * chi_conf
    
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
    
    def unbinding_probs_discrete(self, N_L, N_R, chi_LR):
        """Solve for probabilities p_L and p_R of Ligand/Receptor being UNbound"""
        assert self.binding_model in ( "average", "shaw" ), AssertionError( f'Only usable with average or shaw models' )
        N_L = mp.mpf(N_L)
        N_R = mp.mpf(N_R)
        check = isinstance( chi_LR, np.ndarray)
        
        if check:
            chi_LR = mp.mpf(chi_LR[0])
        else:
            chi_LR = mp.mpf(chi_LR)

        p_L = (N_L-N_R) * chi_LR - 1.0 + mp.sqrt(4.0*N_L*chi_LR + (1.0+(N_R-N_L)*chi_LR)**2)
        p_L /= (2.0*N_L*chi_LR)
        
        p_R = (N_R-N_L) * chi_LR - 1.0 + mp.sqrt(4.0*N_R*chi_LR + (1.0+(N_L-N_R)*chi_LR)**2)
        p_R /= (2.0*N_R*chi_LR)
        
        return p_L, p_R
    
    def unbinding_probs_discrete_positional( self, N_L, sigma_R, r, N, a, K0 ):
        """Solve for probabilities p_L and p_R of Ligand/Receptor being UNbound.
        It assume discrete ligands and a smeared out receptor density, 
        without using any excluded volume effects but correctly accounting for the 
        probability that bonds of different length have different probabilities. We 
        assume a single ligand-receptor pair type.
        """
        assert self.binding_model == "simple", AssertionError( f"Only usable with 'simple' models" )
        R_ee = z_dist = self.r_ee( N, a )
        r_max = self.x_max_span( z_dist = z_dist, 
                                 max_bond_length = 3.0 * R_ee )

        def integrand( r, p_L ):
            if isinstance( p_L, np.ndarray):
                p_L = p_L[0]
            num = 2 * np.pi * p_L * r * sigma_R * self.chi_LR( r, N, a, K0 )
            den = 1.0 + N_L * p_L * self.chi_LR( r, N, a, K0 )
            res = num/den
            return res

        def integral( pL ):
            #integ = [ ( integrand( r, pL ), r ) for r in np.linspace(0, 2 * r_max, 100) ]
            #print( f'XXXXX integ: {integ}' )
            res = np.trapz( [ integrand( r, pL ) for r in np.linspace(0, 2 * r_max, 100) ], np.linspace(0, 2 * r_max, 100) )
            #print( f"YYY res: {res}" )
            #assert isinstance( res, np.float64), AssertionError( f'num: {num} is not a float' )
            return res

        def fun( pL ):
            return ( pL + integral( pL ) - 1.0 )**2
        
        initial_guess = 0.5
        bounds = [(0.0, 1.0)]
        res = minimize( fun, initial_guess, bounds=bounds)
        p_L = res.x[0]
        
        p_R = 1.0 / ( 1.0 + N_L * p_L * self.chi_LR( r, N, a, K0 ) )
        
        return p_L, p_R
    
    def A_bond_discrete(self, N_L, N_R, chi_LR, verbose = False):
        p_L, p_R = self.unbinding_probs_discrete( N_L, N_R, chi_LR )
        if p_L == 0:
            return np.inf
        else:
            bond_energy_L = N_L * ( mp.log(p_L) + 0.5*(1 - p_L))

        if p_R == 0:
            return np.inf
        else:
            bond_energy_R = N_R * (mp.log(p_R) + 0.5*(1 - p_R))
        if verbose:
            print( f'Bond energy: {bond_energy_L + bond_energy_R}' )

        print( f"discrete, pL {p_L}, chi_LR {chi_LR}")
        print( f"discrete, pR {float(p_R):3.5e}")
        print( f'Bond energy - L: {bond_energy_L} R: {bond_energy_R}' )

        return bond_energy_L + bond_energy_R
        
    def A_bond_positional( self, N_L, sigma_R, N, a, K0, verbose = False):
        # Note that this first part is only needed to calculate p_L and the value
        # assumed for r is irrelevant, we could have used any value, here we use
        # r = 1.0
        p_L, p_R = self.unbinding_probs_discrete_positional( N_L, sigma_R, 1.0, N, a, K0 )
        if p_L == 0:
            return np.inf
        else:
            bond_energy_L = N_L * ( mp.log(p_L) + 0.5*(1 - p_L))
            print( f"positional, pL {p_L}")

        if p_R == 0:
            return np.inf
        else:
            r_max = self.x_max_span( z_dist = self.r_ee( N = self.N_long , a = self.a_mono ), 
                                    max_bond_length = 3.0 * self.r_ee( N = self.N_long, a = self.a_mono ) )
            def fun( r, pL ):
                den = 1.0 + N_L * pL * self.chi_LR( r, N, a, K0 )
                p_R = 1.0 / den  
                return 2.0 * mp.pi * r * sigma_R * ( mp.log( p_R ) + 0.5 * (1.0 - p_R ) )
            
            bond_energy_R = quad(fun, 0.0, r_max, args=(p_L) )[ 0 ]
            print( f"positional, chi_LR( min ), chi_LR( max ) {self.chi_LR( 0.0, N, a, K0 )}, {self.chi_LR( r_max, N, a, K0 )}")
            p_R_min = float( 1.0 / (1.0 + N_L * p_L * self.chi_LR( 0, N, a, K0 )) )
            p_R_max = float( 1.0 / (1.0 + N_L * p_L * self.chi_LR( r_max, N, a, K0 )) )
            print( f"positional, p_R( max ) {p_R_max:3.5e} p_R_min {p_R_min:3.5e}")
        print( f'Bond energy - L: {bond_energy_L} R: {bond_energy_R}' )

        if verbose:
            print( f'Bond energy: {bond_energy_L + bond_energy_R}' )

        return bond_energy_L + bond_energy_R
    
    
    def W_bond(self, h, sigma_L, sigma_R, N, a, K0, verbose = False ):
        """Calculate bonding contribution to free energy per unit area"""
        K = self.K_LR(h, N, a, K0)
        p_L, p_R = self.unbinding_probs(sigma_L, sigma_R, K)

        # Free energy calculation per unit area
        if p_L == 0:
            W1 = np.inf
        else:
            W1 = sigma_L * (mp.log(p_L) + 0.5*(1 - p_L))

        if p_R == 0:
            W2 = np.inf
        else:
            W2 = sigma_R * (mp.log(p_R) + 0.5*(1 - p_R))

        res = ( W1 + W2 ) * self.kT
        if verbose:
            #print( f'Bond strength: {K}, p_L: {p_L}, p_R: {p_R}, W1: {W1}, W2: {W2}, bond energy: {res}' )
            print( f'Bond energy density: {res}' )

        return res

    def W_steric(self, h, sigma_polymer, N, a, verbose = False):
        """Calculate steric repulsion free energy per unit area"""
        # This is to avoid numerical problems
        assert self.polymer_model == "gaussian", AssertionError( f"Repulsion implemented only for gaussian polymer")
        check = isinstance( h, np.ndarray)
        
        if check:
            h = mp.mpf(h[0])
        else:
            h = mp.mpf(h)

        if h == 0:
            res = np.inf
        else:
            res = -self.kT * sigma_polymer * mp.log(mp.erf( mp.sqrt( 3 * h**2 / (2 * N * a**2))))
        if verbose:
            print( f'Steric repulsion: {res}' )
        return res
    
    def W_total(self, h, sigma_L, sigma_polymer, sigma_R, N, a, K0, verbose = False):
        """Calculate total interaction free energy per unit area"""
        Ree = self.r_ee( N, self.a_mono )
        W_bond = self.W_bond(h, sigma_L, sigma_R, N, a, K0, verbose)
        W_steric = self.W_steric(h, sigma_polymer, N, a, verbose)
        if verbose:
            print( f'h/Ree {h/Ree} W_bond: {W_bond}, W_steric: {W_steric}' )
        return W_bond + W_steric
    
    def calculate_binding_constant(self, K0, sigma_L, sigma_R,
                                   sigma_polymer = None,
                                   z_max = None, 
                                   verbose = False):
        R_NP = self.R_NP

        if self.binding_model == "shaw":
            """Calculate binding constant using results from Shaw's paper"""
            r_max_3p4K = self.R_max_3p4K 
            r_ee_2K = self.R_ee_2K
            v_bind = r_max_3p4K**3

            A_R, A_P, V_eff = self.geometric_parameters( R_NP = R_NP, 
                                                    R_long = r_max_3p4K, 
                                                    R_short = r_ee_2K )

            N_L = sigma_L * A_P
            N_R = sigma_R * A_R

            chi_LR = K0 / V_eff # 1 / V_eff is an effective density. The lower the harder to bond
            print( f"shaw, chi_LR: {chi_LR}" )
            K_bind = v_bind * mp.exp( -self.A_bond_discrete( N_L, N_R, chi_LR, verbose = verbose ) )

            if verbose:
                print( f"K bind: {K_bind}" )

            return K_bind
        
        elif self.binding_model in ( "simple", "average" ):
            """Calculate binding constant using my initial approximation, as previously done"""
            r_ee_long = self.r_ee( N = self.N_long, a = self.a_mono )
            r_ave_bond = self.r_ave_bond( z_dist = r_ee_long, max_bond_length = 3.0 * r_ee_long ) 
            v_bind = r_ee_long**3

            _, A_P, _ = self.geometric_parameters( R_NP = R_NP, 
                                                    R_long = 3 * r_ee_long, 
                                                    R_short = r_ee_long )
            
            N_L = sigma_L * A_P

            x_max = self.x_max_span( z_dist = r_ee_long, max_bond_length = 3.0 * r_ee_long )
            A_R = np.pi * x_max**2 # Approximate area spanned by a ligand, calculated assuming max length is 3 times average
                                  # R_ee o the polymer holding the ligand
            N_R = sigma_R * A_R

            print( f'Active number of ligands and receptors in binding zone: N_L: {N_L}, N_R: {N_R}' )

            v_bind = r_ee_long**3
            if self.binding_model == "average":
                chi_LR = self.chi_LR( r_bond = r_ave_bond, 
                                     N = self.N_long,
                                     a = self.a_mono,
                                     K0 = K0 )
                K_bind = v_bind * mp.exp( -self.A_bond_discrete( N_L = N_L, 
                                                                N_R = N_R, 
                                                                chi_LR = chi_LR, 
                                                                verbose = verbose ) )
            elif self.binding_model == "simple":
                K_bind = v_bind * mp.exp( -self.A_bond_positional( N_L = N_L, 
                                                                  sigma_R = sigma_R, 
                                                                  N = self.N_long, 
                                                                  a = self.a_mono, 
                                                                  K0 = K0, 
                                                                  verbose = verbose ) )
            return K_bind
    
        elif self.binding_model == "saddle":
            z_max = self.N_long * self.a_mono
            #"""Calculate binding constant using Derjaguin approximation AND saddle point approximation"""
            def force(h):
                return 2 * np.pi * R_NP * self.W_total(h, sigma_L = sigma_L, 
                                                       sigma_polymer = sigma_polymer, 
                                                       sigma_R = sigma_R, 
                                                       N = self.N_long, 
                                                       a = self.a_mono, 
                                                       K0 = K0, 
                                                       verbose = verbose 
                                                       )
        
            # Find equilibrium binding distance where W_total = 0
            def find_z_bind(h):
                return self.W_total(h, sigma_L = sigma_L, 
                                    sigma_polymer = sigma_polymer, 
                                    sigma_R = sigma_R, 
                                    N = self.N_long, 
                                    a = self.a_mono, 
                                    K0 = K0, 
                                    verbose = verbose 
                                    )

            R_ee = self.r_ee( N = self.N_long, a = self.a_mono ) 

            initial_guess = R_ee
            bounds = [(0.0, z_max)]
            result = minimize(find_z_bind, initial_guess, bounds=bounds)
            z_bind = result.x[ 0 ]
        
            if z_bind > z_max:
                raise ValueError( f'Equilibrium binding distance is too large z/R_ee: {z_bind/R_ee}' )

            if verbose:
                print( f'Equilibrium binding distance, normalized to Ree: {z_bind / R_ee }' )       

            # Calculate the second derivative of A at minimum. This is minus the first derivative of the force
            dh = 1e-8  # Small step for numerical derivative
            F_prime = -(force(z_bind + dh) - force(z_bind - dh))/(2*dh)
            print( f'Distance minimising plane-plane interaction (z_bind/R_ee): {z_bind/R_ee}' )
            print( f'Second derivative at minimum: {F_prime}' )
        
            # Binding constant using saddle point approximation
            Area = mp.pi * self.N_long * self.a_mono**2  # Approximate area spanned by ligand
            energy_min = np.trapz([force(h) for h in np.linspace(z_bind, z_max, 100)],
                                  np.linspace(z_bind, z_max, 100))
        
            if verbose:
                print( f'Energy minimum: {energy_min}' )
                print( f'Second derivative at minimum: {F_prime}' )

            K_bind = Area * mp.exp(-energy_min/self.kT) * mp.sqrt(2*mp.pi/(F_prime/self.kT))
            return K_bind
        
        elif self.binding_model == "exact":
            #"""Calculate binding constant using Derjaguin approximation"""
            z_max = self.N_long * self.a_mono
            N = self.N_long
            a = self.a_mono
            def force(h):
                W_total = self.W_total(h, sigma_L = sigma_L, 
                                                       sigma_polymer = sigma_polymer, 
                                                       sigma_R = sigma_R, 
                                                       N = N, 
                                                       a = a, 
                                                       K0 = K0, 
                                                       verbose = verbose 
                                                       )
                if verbose:
                    print( f'W_total: {W_total} (kbT/nm^2)' )
                    print( f'Force: {2 * mp.pi * R_NP * W_total} (kbT/nm)' )
                return 2 * mp.pi * R_NP * W_total

            def A(h):
                force_values = [force(x) for x in np.linspace(h, z_max, 100)]
                x_values = np.linspace(h, z_max, 100)
                Ah = np.trapz(force_values, x_values )
                minAh = np.min(force_values) * ( z_max - h )
                assert Ah >= minAh, AssertionError( f'This should not happen: Ah: {Ah} < minAh: {minAh}' )
                if verbose:
                    print( f'h {h}, A(h) {Ah}' )

                return Ah
        
            def integrand(h):
                return mp.exp(-A(h)/self.kT)
        
            A_L = mp.pi * N * a**2  # Approximate area spanned by ligand
        
            K_bind = A_L *  np.trapz([integrand(h) for h in np.linspace(0, z_max, 100)],
                                    np.linspace(0, z_max, 100))
            return K_bind
    
    def calculate_bound_fraction(self, K_bind, verbose = False):
        A_cell = self.A_cell
        NP_conc = self.NP_conc
        cell_conc = self.cell_conc
        R_NP = self.R_NP
        """Calculate fraction of bound nanoparticles"""
        if K_bind == np.inf:
            return 1.0
        else:
            M_conc = (A_cell/(np.pi * R_NP**2)) * cell_conc
        
            term = (NP_conc + M_conc) * K_bind + 1 
            sqrt_term = mp.sqrt( mp.power( term, 2 ) - 4 * NP_conc * M_conc * K_bind**2 )
        
            NP_M_conc = (term - sqrt_term)/(2 * K_bind)

            bound_fraction = NP_M_conc / NP_conc

            if verbose:
                print( f'term: {term}, sqrt_term: {sqrt_term}, NP_M_conc: {NP_M_conc}' )
                print( f'M concentration / NP_conc: {M_conc/NP_conc}, bound fraction: {bound_fraction}' )
        
            return bound_fraction
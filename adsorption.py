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
        #print( f'1) The values of R_max_3p4K and R_ee_2K are taken at different grafting distances' )
        #print( f'This problem might be solvable using a Komura-Safran model for the bimodal brush' )
        #print( f"""2) Brush repulsion only counted for long brush in Gaussian state but could be improved
        #       - and you know how using KS-blob model. However, this should not really matter IF short brush is
        #      very hard? Check""" )
        #print( f'3) Bond energy using Flory model + Gaussian approx around a different average value gives quadratic theory again, nice!' )

    def K_LR(self, h, N, a, K_bind_0):
        """Calculate area-weighted average bond strength K_LR(h) between ligand-receptor pairs,
        assuming the planes containing ligands and receptors are parallel and at a distance 'h'
        from each other.
        K_LR is calculated assuming receptors are fixed point on a surface and ligands are tethered to 
        a NP by a Gaussian polymer chain"""
        if self.polymer_model in [ "gaussian", "Flory" ]:
            # Note that even in the Flory model, close to its minimum the behaviour will be quadratic/Gaussian, albeit
            # shifted at a different average which is that given by the Flory exponent for a self-avoiding walk
            r_ee2 = self.r_ee( N, a )**2
            #prefactor = K_bind_0 * np.sqrt( 12.0 / ( np.pi * N * a**2)) 
            #exp_term = np.exp(-3.0 * h / (4 * N * a**2))
            #erf_num = erf(np.sqrt(3 * h**2/(4 * N * a**2)))
            #erf_den = erf(np.sqrt(3 * h**2/(2 * N * a**2)))
            
            prefactor = K_bind_0 * np.sqrt( 12.0 / ( np.pi * r_ee2)) 
            exp_term = np.exp(-3.0 * h**2 / (4 * r_ee2))
            erf_num = erf(np.sqrt(3 * h**2/(4 * r_ee2)))
            erf_den = erf(np.sqrt(3 * h**2/(2 * r_ee2)))

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
    
    def x_max_span( self, z_dist, max_bond_length = None ):
        """The radius of the maximum circle spanned by a ligand on the receptor-containing surface.
        Same assumption/geoemtry as above"""
        if max_bond_length is None:
            max_bond_length = 3.0 * z_dist
        assert max_bond_length > z_dist # Sanity check
        x_max = np.sqrt( max_bond_length**2 - z_dist**2 ) 
        return x_max 
    
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
    
    def W_bond(self, h, sigma_L, sigma_R, N, a, K_bind_0, verbose = False ):
        """Calculate bonding contribution to free energy per unit area"""
        K = self.K_LR(h, N, a, K_bind_0)
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
        #print( f'h/Ree {h/np.sqrt(N)*a} R_ee {np.sqrt(N)* a}, Bond energy density: {res}' )
        if verbose:
            print( f'Bond energy density: {res}' )

        return res

    def W_steric(self, h, sigma_polymer, N, a, verbose = False):
        """Calculate steric repulsion free energy per unit area"""
        # This is to avoid numerical problems
        assert self.polymer_model in [ "gaussian", "Flory" ], AssertionError( f"Repulsion implemented only for gaussian polymer")
        check = isinstance( h, np.ndarray)
        
        if check:
            h = mp.mpf(h[0])
        else:
            h = mp.mpf(h)

        if h == 0:
            res = np.inf
        else:
            res = -self.kT * sigma_polymer * mp.log(mp.erf( mp.sqrt( 3 * h**2 / (2 * N * a**2))))
            #print( f'h/Ree {h/np.sqrt(N)*a} R_ee {np.sqrt(N)* a}, Repulsive energy density: {res}' )
        if verbose:
            print( f'Steric repulsion: {res}' )
        return res
    
    def W_total(self, h, sigma_L, sigma_polymer, sigma_R, N, a, K_bind_0, verbose = False):
        """Calculate total interaction free energy per unit area"""
        Ree = self.r_ee( N, self.a_mono )
        W_bond = self.W_bond(h, sigma_L, sigma_R, N, a, K_bind_0, verbose)
        W_steric = self.W_steric(h, sigma_polymer, N, a, verbose)
        if verbose:
            print( f'h/Ree {h/Ree} W_bond: {W_bond}, W_steric: {W_steric}' )
        return W_bond + W_steric
    
    def calculate_binding_constant(self, K_bind_0, sigma_L, sigma_R,
                                   sigma_polymer = None,
                                   z_max = None, 
                                   verbose = False):
        R_NP = self.R_NP

        if self.binding_model == "saddle":
            z_max = self.N_long * self.a_mono
            #"""Calculate binding constant using Derjaguin approximation AND saddle point approximation"""
            def force(h):
                return 2 * np.pi * R_NP * self.W_total(h, sigma_L = sigma_L, 
                                                       sigma_polymer = sigma_polymer, 
                                                       sigma_R = sigma_R, 
                                                       N = self.N_long, 
                                                       a = self.a_mono, 
                                                       K_bind_0 = K_bind_0, 
                                                       verbose = verbose 
                                                       )
        
            # Find equilibrium binding distance where W_total = 0
            def find_z_bind(h):
                return self.W_total(h, sigma_L = sigma_L, 
                                    sigma_polymer = sigma_polymer, 
                                    sigma_R = sigma_R, 
                                    N = self.N_long, 
                                    a = self.a_mono, 
                                    K_bind_0 = K_bind_0, 
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
                print( f'Equilibrium binding distance, normalized to max linear extension: {z_bind / z_max }' )       

            # Calculate the second derivative of A at minimum. This is minus the first derivative of the force
            dh = 1e-8  # Small step for numerical derivative
            F_prime = -(force(z_bind + dh) - force(z_bind - dh))/(2*dh)
            print( f'Distance minimising plane-plane interaction (z_bind/R_ee): {z_bind/R_ee}' )
            print( f'Distance minimising plane-plane interaction (z_bind/z_max): {z_bind/z_max}' )
            try:
                assert F_prime >= 0.0, AssertionError( f'Second derivative at minimum: {F_prime} should not be negative' )
            except:
                K_bind = np.inf
                return K_bind
        
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
                                                       K_bind_0 = K_bind_0, 
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

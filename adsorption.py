import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve, minimize
from mpmath import mp

class MultivalentBinding:
    def __init__(self, kT=1.0):
        self.kT = kT  # Thermal energy (kB*T)
        self.nm = 1.0 # Sets the units of length
        self.nm2 = ( self.nm )**2 # Sets the units of area
        self.nm3 = ( self.nm )**3 # Sets the units of volume
        self.rhostd = 6.023e23/ ( 1e24 * self.nm3 ) 
        
    def K_LR(self, h, N, a, K0):
        """Calculate average bond strength K_LR(h) between ligand-receptor pairs"""
        prefactor = K0 * np.sqrt(12/(np.pi * N * a**2)) 
        exp_term = np.exp(-3 * h / (4 * N * a**2))
        erf_num = erf(np.sqrt(3 * h**2/(4 * N * a**2)))
        erf_den = erf(np.sqrt(3 * h**2/(2 * N * a**2)))

        #print( f'K0 {K0}, prefactor {prefactor}, exp_term {exp_term}, erf_num {erf_num}, erf_den {erf_den}' )

        if h == 0:
            return prefactor / np.sqrt(2.0)
        else:
            return prefactor * exp_term * erf_num/erf_den
        
    
    def unbinding_probs(self, sigma_L, sigma_R, K_LR):
        """Solve for probabilities p_L and p_R of Ligand/Receptor being UNbound"""
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
    
    def unbinding_probs_low_precision(self, sigma_L, sigma_R, K_LR):
        """Solve for probabilities p_L and p_R of Ligand/Receptor being UNbound.
        This is a low precision version of the unbinding_probs method, and gets
        numerical errors due to difference between small numbers for very large
        and very small values of K_LR"""
        
        p_L = (sigma_L-sigma_R) * K_LR - 1 + np.sqrt(4*sigma_L*K_LR + (1+(sigma_R-sigma_L)*K_LR)**2)
        p_L /= (2*sigma_L*K_LR)
        
        p_R = (sigma_R-sigma_L) * K_LR - 1 + np.sqrt(4*sigma_R*K_LR + (1+(sigma_L-sigma_R)*K_LR)**2)
        p_R /= (2*sigma_R*K_LR)
        
        return p_L, p_R
    
    
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
        Ree = mp.sqrt( N * a**2 )
        W_bond = self.W_bond(h, sigma_L, sigma_R, N, a, K0, verbose)
        W_steric = self.W_steric(h, sigma_polymer, N, a, verbose)
        if verbose:
            print( f'h/Ree {h/Ree} W_bond: {W_bond}, W_steric: {W_steric}' )
        return W_bond + W_steric
    
    def calculate_binding_constant_shaw(self, R_NP, sigma_L, sigma_R, Nlong, Nshort a, K0, z_max, verbose = False):
        """Calculate binding constant using my initial approximation, as previously done"""
        Rlong = np.sqrt( Nlong ) * a
        Rshort = np.sqrt( Nshort ) * a
        f1 = 1.0 - ((R_NP/2) + Rshort ) / ((R_NP/2) + Rlong )
        A_p = np.pi * (R_NP/2)**2 / 2.0 * f1
        NL = sigma_L * A_p
        A_R = np.pi * ( RNP/2 + Rlong )**2 - ( RNP/2 + Rshort )**2
        NR = sigma_R * A_R

        return K_bind
    
    def calculate_binding_constant_approx(self, R_NP, sigma_L, sigma_polymer, sigma_R, N, a, K0, z_max, verbose = False):
        """Calculate binding constant using my initial approximation, as previously done"""
        return K_bind
    
    def calculate_binding_constant_saddle(self, R_NP, sigma_L, sigma_polymer, sigma_R, N, a, K0, z_max, verbose = False):
        """Calculate binding constant using Derjaguin approximation AND saddle point approximation"""
        def force(h):
            return 2 * np.pi * R_NP * self.W_total(h, sigma_L, sigma_polymer, sigma_R, N, a, K0, verbose = verbose )
        
        # Find equilibrium binding distance where W_total = 0
        def find_z_bind(h):
            return self.W_total(h, sigma_L, sigma_polymer, sigma_R, N, a, K0)
        
        initial_guess = np.sqrt(N) * a**2
        bounds = [(0.0, z_max)]
        result = minimize(find_z_bind, initial_guess, bounds=bounds)
        z_bind = result.x[ 0 ]
        
        if z_bind > z_max:
            Ree = np.sqrt( N ) * a
            raise ValueError( f'Equilibrium binding distance is too large z/Ree: {z_bind/Ree}' )

        if verbose:
            Ree = np.sqrt( N ) * a
            print( f'Equilibrium binding distance, normalized to Ree: {z_bind / Ree }' )       

        # Calculate the second derivative of A at minimum. This is minus the first derivative of the force
        dh = 1e-8  # Small step for numerical derivative
        F_prime = -(force(z_bind + dh) - force(z_bind - dh))/(2*dh)
        print( f'Second derivative at minimum: {F_prime}' )
        
        # Binding constant using saddle point approximation
        A_L = mp.pi * N * a**2  # Approximate area spanned by ligand
        energy_min = np.trapz([force(h) for h in np.linspace(z_bind, z_max, 100)],
                              np.linspace(z_bind, z_max, 100))
        
        if verbose:
            print( f'Energy minimum: {energy_min}' )
            print( f'Second derivative at minimum: {F_prime}' )

        K_bind = A_L * mp.exp(-energy_min/self.kT) * mp.sqrt(2*mp.pi/(F_prime/self.kT))
        return K_bind
    
    def calculate_binding_constant(self, R_NP, sigma_L, sigma_polymer, sigma_R, N, a, K0, z_max, verbose=False):
        """Calculate binding constant using Derjaguin approximation"""
        def force(h):
            W_total = self.W_total(h, sigma_L, sigma_polymer, sigma_R, N, a, K0, verbose = verbose )
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
    
    def calculate_bound_fraction(self, K_bind, NP_conc, cell_conc, R_NP, A_cell, verbose = False):
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
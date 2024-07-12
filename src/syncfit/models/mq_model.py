'''
Implementation of the Margalit & Quataert (2024) Thermal Electron Model

Much of this code relies on the thermalsyn_v2 module provided by the Margalit &
Quataert (2024) paper.
'''

import numpy as np
from .syncfit_model import SyncfitModel
from .thermalsyn_v2 import Lnu_of_nu
from astropy import units as u
from astropy import constants as c

class MQModel(SyncfitModel):
    
    def get_labels(p=None):
        if p is None:
            return ['p', 'bG_sh', 'log Mdot', 'log_epsilon_T', 'log_epsilon_e', 'log_epsilon_B']
        else:
            return ['bG_sh', 'log Mdot', 'log_epsilon_T', 'log_epsilon_e', 'log_epsilon_B']

    def SED(nu, p, log_bG_sh, logMdot, log_epsilon_T, log_epsilon_e, log_epsilon_B,
            lum_dist, t, **kwargs):       

        # set microphysical and geometric parameters
        # log_epsilon_e = -1
        # log_epsilon_B = log_epsilon_e # assume equipartition
        delta = 10**log_epsilon_e/10**log_epsilon_T
        f = 3.0/16.0
        ell_dec = 1.0

        Mdot_over_vw = (10**logMdot*(c.M_sun/u.yr/1e8)).cgs.value

        Lnu = Lnu_of_nu(
            10**log_bG_sh, Mdot_over_vw, nu, t, p=p, 
            epsilon_T=10**log_epsilon_T, epsilon_B=10**log_epsilon_B, epsilon_e=10**log_epsilon_e,
            f=f,ell_dec=ell_dec,radius_insteadof_time=False
        ) * u.erg / (u.s * u.Hz)

        lum_dist_cm = lum_dist*u.cm # give it units so the conversion works well
        Fnu = (Lnu / (4*np.pi*(lum_dist_cm)**2)).to(u.mJy) # mJy

        return Fnu.value
    
    def lnprior(theta, nu, F, p=None, **kwargs):
        '''
        The prior
        '''

        if p is None:
            p, bG_sh, logMdot, epsilon_T, epsilon_e, log_epsilon_B = theta
        else:
            bG_sh, logMdot, epsilon_T, epsilon_e, log_epsilon_B = theta

        epsilon_B = epsilon_e # assume equipartition

        if (2 < p < 4 and 
            0.01 < bG_sh < 100 and 
            -10 < logMdot < 0 and 
            -6 < epsilon_e < 0 and 
            -6 < epsilon_T < 0 and
            -6 < epsilon_B < 0 and
            0 <= 10**epsilon_e + 10**epsilon_B + 10**epsilon_T <= 1):

            return 0.0
        else:
            return -np.inf

    def dynesty_transform(theta, nu, F, upperlimit, p=None, **kwargs):
        '''
        Prior transform function for dynesty

        theta is expected to be a 1D array with each value in the range [0,1)
        so we need to transform each to parameter space
        '''
        
        if p is None:
            fixed_p = False
            p, log_bG_sh, logMdot, epsilon_T, epsilon_e, epsilon_B = theta
        else:
            fixed_p = True
            log_bG_sh, logMdot, epsilon_T, epsilon_e, epsilon_B = theta

        # log_bG_sh should be between -2 and 2
        log_bG_sh = log_bG_sh*4 - 2

        # -10 < logMdot < 0
        logMdot*=-10

        # -3 < epsilon_e < 0
        epsilon_e*=-6
        epsilon_B*=-6
    
        # -3 < epsilon_T < 0
        epsilon_T*=-6
        
        if not fixed_p:
            # p should be between 2 and 4
            p = 2*p + 2
            
            # p between 2.5 and 3.5, let's be a little more restrictive
            #p += 2.5
            
            return p,log_bG_sh,logMdot,epsilon_T,epsilon_e, epsilon_B
        return log_bG_sh,logMdot,epsilon_T,epsilon_e, epsilon_B

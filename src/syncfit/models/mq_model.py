'''
Implementation of the Margalit & Quataert (2024) Thermal Electron Model

Much of this code relies on the thermalsyn_v2 module provided by the Margalit &
Quataert (2024) paper.
'''

import numpy as np
from .syncfit_model import SyncfitModel
from .thermal_util import Lnu_of_nu
from astropy import units as u
from astropy import constants as c

class MQModel(SyncfitModel):

    def __init__(self, p=None, t=None):
        # set the default prior for this model
        self.prior = dict(
            log_bG_sh=[-3,3],
            log_Mdot=[-10,0],
            log_epsilon_e=[-3,0],
            log_epsilon_B=[-3,0],
            log_epsilon_T=[-3,0]
        )

        if p is None:
            self.prior['p'] = [2,4]

        if t is None:
            self.prior['t'] = [0.1, 1_000] # super wide prior as default

        # then initiate the superclass
        super().__init__(self.prior, p=p)
        self.t = t
                                
    def SED(self, nu, p, log_bG_sh, logMdot, log_epsilon_T, log_epsilon_e, log_epsilon_B,
            t, lum_dist, **kwargs):       

        # set microphysical and geometric parameters
        # log_epsilon_e = -1
        # log_epsilon_B = log_epsilon_e # assume equipartition
        delta = 10**log_epsilon_e/10**log_epsilon_T
        f = 3.0/16.0
        ell_dec = 1.0

        t = (t*u.day).to(u.s).value
        
        Mdot_over_vw = (10**logMdot*(c.M_sun/u.yr/1e8)).cgs.value

        Lnu = Lnu_of_nu(
            10**log_bG_sh, Mdot_over_vw, nu, t, p=p, 
            epsilon_T=10**log_epsilon_T, epsilon_B=10**log_epsilon_B, epsilon_e=10**log_epsilon_e,
            f=f,ell_dec=ell_dec,radius_insteadof_time=False
        ) * u.erg / (u.s * u.Hz)

        lum_dist_cm = lum_dist*u.cm # give it units so the conversion works well
        Fnu = (Lnu / (4*np.pi*(lum_dist_cm)**2)).to(u.mJy) # mJy

        return Fnu.value

    def lnprior(self, theta, nu, F, upperlimit, **kwargs):
        '''
        Logarithmic prior function that can be changed based on the SED model.
        '''
        uppertest = SyncfitModel._is_below_upperlimits(
            nu, F, upperlimit, theta, self.SED
        )

        packed_theta = self.pack_theta(theta)
        
        all_res = []
        for param, val in self.prior.items():
            res = val[0] < packed_theta[param] < val[1]
            all_res.append(res)
            
        if (all(all_res) and
            uppertest and
            0 <= 10**packed_theta['log_epsilon_e'] + 10**packed_theta['log_epsilon_B'] + 10**packed_theta['log_epsilon_T'] <= 1
            ):
            return 0.0
        else:
            return -np.inf

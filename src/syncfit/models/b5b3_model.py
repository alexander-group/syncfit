'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B5B3(SyncfitModel):
    '''
    Two-break model that uses both the self-absorption break and the cooling break.
    This model forces the cooling break to always be larger than the self-absorption
    break.
    '''

    def __init__(self, p=None):
        # then set the default prior for this model
        if p is None:
            self.prior = dict(
                p=[2,4],
                log_F_nu=[-4,2],
                log_nu_a=[6,11],
                log_nu_c=[7,15]
            )
        else:
            self.prior = dict(
                log_F_nu=[-4,2],
                log_nu_a=[6,11],
                log_nu_c=[7,15]
            )

        super().__init__(self.prior, p=p)
        
    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, log_nu_c, **kwargs):
        b1 = 5/2
        b2 = (1-p)/2
        b3 = -p/2

        s12 = 0.8-0.03*p
        s23 = 1.15-0.06*p

        F_nu = 10**log_F_nu
        nu_c = 10**log_nu_c
        nu_a = 10**log_nu_a

        term1 = ((nu/nu_a)**(-s12*b1) + (nu/nu_a)**(-s12*b2))**(-1/s12)
        term2 = (1 + (nu/nu_c)**(s23*(b2-b3)))**(-1/s23)

        return F_nu * term1 * term2

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
            packed_theta['log_nu_c'] > packed_theta['log_nu_a']
            ):
            return 0.0
        else:
            return -np.inf

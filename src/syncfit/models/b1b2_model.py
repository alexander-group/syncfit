'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B1B2(SyncfitModel):
    '''
    Two-break model for the self-absorption break (nu_a) and the minimal energy break
    (nu_m). This model uses nu_m > nu_a, the opposite of the B4B5 model.
    '''

    def __init__(self, p=None):
        # then set the default prior for this model
        if p is None:
            self.prior = dict(
                p=[2,4],
                log_F_nu=[-4,2],
                log_nu_a=[6,11],
                log_nu_m=[7,15]
            )
        else:
            self.prior = dict(
                log_F_nu=[-4,2],
                log_nu_a=[6,11],
                log_nu_m=[7,15]
            )

        super().__init__(p=p)

    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, log_nu_m, **kwargs):
        b1 = 2
        b2 = 1/3
        b3 = (1-p)/2

        s1 = 1.06
        s2 = 1.76-0.38*p

        F_nu = 10**log_F_nu
        nu_m = 10**log_nu_m
        nu_a = 10**log_nu_a

        term1 = ( (nu/nu_a)**(-s1*b1) + (nu/nu_a)**(-s1*b2) )**(-1/s1)
        term2 = ( 1 + (nu/nu_m)**(s2*(b2-b3)) )**(-1/s2)

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
            packed_theta['log_nu_m'] > packed_theta['log_nu_a']
            ):
            return 0.0
        else:
            return -np.inf

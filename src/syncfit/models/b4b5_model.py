'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B4B5(SyncfitModel):
    '''
    Two-break model for a combination of the self-absorption break (nu_a) and the
    minimal energy break (nu_m). This model requires that nu_m < nu_a, you should
    subclass this class and overwrite lnprior to redefine this. 
    '''

    def __init__(self, prior=None, p=None):
        # then set the default prior for this model
        if prior is None:
            if p is None:
                self.prior = dict(
                    p=[2,4],
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11],
                    log_nu_m=[-np.inf,6]
                )
            else:
                self.prior = dict(
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11],
                    log_nu_m=[-np.inf,6]
                )
        else:
            self.prior = prior


        super().__init__(self.prior, p=p)
    
    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, log_nu_m, **kwargs):
        b1 = 2
        b2 = 5/2
        b3 = (1-p)/2

        s4 = 3.63*p-1.6
        s2 = 1.25-0.18*p

        F_nu = 10**log_F_nu
        nu_m = 10**log_nu_m
        nu_a = 10**log_nu_a

        term1 = ( (nu/nu_m)**(b1) * np.exp(-s4*(nu/nu_m)**(2/3)) + (nu/nu_m)**(b2))
        term2 = ( 1 + (nu/nu_a)**(s2*(b2-b3)) )**(-1/s2)

        return F_nu * term1 * term2

    def lnprior(self, theta, nu, F, upperlimit, **kwargs):
        '''
        Logarithmic prior function that can be changed based on the SED model.
        '''
        uppertest = SyncfitModel._is_below_upperlimits(
            nu, F, upperlimit, theta, self.SED, **kwargs
        )

        packed_theta = self.pack_theta(theta)
        
        all_res = []
        for param, val in self.prior.items():
            res = val[0] < packed_theta[param] < val[1]
            all_res.append(res)
            
        if (all(all_res) and
            uppertest and
            packed_theta['log_nu_a'] > packed_theta['log_nu_m']
            ):
            return 0.0
        else:
            return -np.inf

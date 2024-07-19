'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B5(SyncfitModel):
    '''
    Single break model for just the self-absorption break.
    '''

    def __init__(self, prior=None, p=None):
        # then set the default prior for this model
        if prior is None:
            if p is None:
                self.prior = dict(
                    p=[2,4],
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11]
                )
            else:
                self.prior = dict(
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11]
                )
        else:
            self.prior = prior

        super().__init__(self.prior, p=p)
        
    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, **kwargs):
        b1 = 5/2
        b2 = (1-p)/2
        s = 1.25-0.18*p

        F_nu = 10**log_F_nu
        nu_a = 10**log_nu_a

        term = ((nu/nu_a)**(-s*b1)+(nu/nu_a)**(-s*b2))

        return F_nu*term**(-1/s)

'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B1B2_B3B4_Weighted(SyncfitModel):
    '''
    This is a specialized model that uses a weighted combination of the B1B2 model and
    the B3B4 model. The idea of this model is from XXXYXYXYX et al. (YYYY).
    '''

    def __init__(self, p=None):
        # then set the default prior for this model
        if p is None:
            self.prior = dict(
                p=[2,4],
                log_F_nu=[-4,2],
                log_nu_a=[6,12],
                log_nu_m=[6,12]
            )
        else:
            self.prior = dict(
                log_F_nu=[-4,2],
                log_nu_a=[6,12],
                log_nu_m=[6,12]
            )

        super().__init__(self.prior, p=p)
            
    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, log_nu_m, **kwargs):
        ### Spectrum 1
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

        F1 = F_nu * term1 * term2

        ### Spectrum 2
        b1 = 2
        b2 = 5/2
        b3 = (1-p)/2

        s4 = 3.63*p-1.6
        s2 = 1.25-0.18*p


        term1 = ( (nu/nu_m)**(b1) * np.exp(-s4*(nu/nu_m)**(2/3)) + (nu/nu_m)**(b2))
        term2 = ( 1 + (nu/nu_a)**(s2*(b2-b3)) )**(-1/s2)

        F2 = F_nu * term1 * term2

        ### Spectrum 1 and 2 weighted
        w1 = (nu_m/nu_a)**2
        w2 = (nu_a/nu_m)**2

        F = (w1*F1+w2*F2) / (w1+w2)

        return F

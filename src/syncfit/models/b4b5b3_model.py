'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B4B5B3(SyncfitModel):
    '''
    Three-break model using the self-absorption break (nu_a), cooling break (nu_c),
    and minimum energy break (nu_m). This model always requires that nu_m < nu_a < nu_c. 
    '''

    def __init__(self, prior=None, p=None):
        # then set the default prior for this model
        if prior is None:
            if p is None:
                self.prior = dict(
                    p=[2,4],
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11],
                    log_nu_m=[0, 6],
                    log_nu_c=[7,15]
                )
            else:
                self.prior = dict(
                    log_F_nu=[-4,2],
                    log_nu_a=[6,11],
                    log_nu_m=[0, 6],
                    log_nu_c=[7,15]
                )
        else:
            self.prior = prior
                
        super().__init__(self.prior, p=p)
            
    # the model, must be named SED!!!
    def SED(self, nu, p, log_F_nu, log_nu_a, log_nu_m, log_nu_c, **kwargs):
        b1 = 2
        b2 = 5/2
        b3 = (1-p)/2
        b4 = -p/2

        s1 = 3.63*p-1.6
        s2 = 1.25-0.18*p
        s3 = 10

        F_nu = 10**log_F_nu
        nu_m = 10**log_nu_m
        nu_a = 10**log_nu_a
        nu_c = 10**log_nu_c


        term1 = ( (nu/nu_m)**(b1) * np.exp(-s1*(nu/nu_m)**(2/3)) + (nu/nu_m)**(b2))
        term2 = ( 1 + (nu/nu_a)**(s2*(b2-b3)) )**(-1/s2)
        term3 = ( 1 + (nu/nu_c)**(s3*(b3-b4)) )**(-1/s3)

        return F_nu * term1 * term2 * term3

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
            packed_theta['log_nu_c'] > packed_theta['log_nu_a'] and
            packed_theta['log_nu_m'] < packed_theta['log_nu_a']
            ):
            return 0.0
        else:
            return -np.inf

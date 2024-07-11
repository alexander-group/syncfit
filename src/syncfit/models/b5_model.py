'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .syncfit_model import SyncfitModel

class B5(SyncfitModel):
    '''
    Single break model for just the self-absorption break.
    '''

    def get_labels(p=None):
        if p is None:
            return ['p','log F_v', 'log nu_a']
        else:
            return ['log F_v', 'log nu_a']

    # the model, must be named SED!!!
    def SED(nu, p, log_F_nu, log_nu_a, **kwargs):
        b1 = 5/2
        b2 = (1-p)/2
        s = 1.25-0.18*p

        F_nu = 10**log_F_nu
        nu_a = 10**log_nu_a

        term = ((nu/nu_a)**(-s*b1)+(nu/nu_a)**(-s*b2))

        return F_nu*term**(-1/s)

    def lnprior(theta, nu, F, upperlimit, p=None, **kwargs):
        ''' Priors: '''
        uppertest = SyncfitModel._is_below_upperlimits(
            nu, F, upperlimit, theta, B5.SED, p=p
        )
        
        if p is None:
            p, log_F_nu, log_nu_a = theta
        else:
            log_F_nu, log_nu_a = theta

        if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 11 and uppertest:
            return 0.0

        else:
            return -np.inf    


    def dynesty_transform(theta, nu, F, upperlimit, p=None, **kwargs):
        '''
        Prior transform for dynesty
        '''
        
        if p is None:
            p, log_F_nu, log_nu_a = theta
            fixed_p = False
        else:
            fixed_p = True
            log_F_nu, log_nu_a = theta


        # log_F_nu between -4 and 2
        log_F_nu = log_F_nu*6 - 4
        
        # log_nu_a between 6 and 11
        log_nu_a = log_nu_a*5 + 6
        

        if not fixed_p:
            # p should be between 2 and 4
            p = 2*p + 2
            
            return p,log_F_nu,log_nu_a
        return log_F_nu,log_nu_a
            

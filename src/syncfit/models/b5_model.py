'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .base_model import BaseModel

class B5(BaseModel):
    '''
    Single break model for just the self-absorption break.
    '''

    def get_labels(p=None):
        if p is None:
            return ['p','log F_v', 'log nu_a']
        else:
            return ['log F_v', 'log nu_a']

    # the model, must be named SED!!!
    def SED(nu, p, log_F_nu, log_nu_a):
        b1 = 5/2
        b2 = (1-p)/2
        s = 1.25-0.18*p

        F_nu = 10**log_F_nu
        nu_a = 10**log_nu_a

        term = ((nu/nu_a)**(-s*b1)+(nu/nu_a)**(-s*b2))

        return F_nu*term**(-1/s)

    def lnprior(theta, p=None, **kwargs):
        ''' Priors: '''
        if p is None:
            p, log_F_nu, log_nu_a= theta
        else:
            log_F_nu, log_nu_a = theta
        if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 11:
            return 0.0

        else:
            return -np.inf    

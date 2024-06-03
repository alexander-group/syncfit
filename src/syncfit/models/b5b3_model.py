'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .base_model import BaseModel

class B5B3(BaseModel):
    '''
    Two-break model that uses both the self-absorption break and the cooling break.
    This model forces the cooling break to always be larger than the self-absorption
    break.
    '''

    # Write some getters for things that are model specific
    def get_labels(p=None):
        if p is None:
            return ['p','log F_v', 'log nu_a','log nu_c']
        else:
            return ['log F_v', 'log nu_a','log nu_c']

    # the model, must be named SED!!!
    def SED(nu, p, log_F_nu, log_nu_a, log_nu_c):
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

    def lnprior(theta, p=None, **kwargs):
        ''' Priors: '''
        if p is None:
            p, log_F_nu, log_nu_a, log_nu_c = theta
        else:
            log_F_nu, log_nu_a, log_nu_c = theta

        if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 11 and log_nu_c > log_nu_a:
            return 0.0

        else:
            return -np.inf

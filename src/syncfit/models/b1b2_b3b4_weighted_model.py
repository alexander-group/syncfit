'''
Various models to use in MCMC fitting 
'''
import numpy as np
from .base_model import BaseModel

class B1B2_B3B4_Weighted(BaseModel):

    def get_labels(p=None):
        if p is None:
            return ['p','log F_v', 'log nu_a','log nu_m']
        else:
            return ['log F_v', 'log nu_a','log nu_m']

    # the model, must be named SED!!!
    def SED(nu, p, log_F_nu, log_nu_a, log_nu_m):
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

    def lnprior(theta, p=None, **kwargs):
        ''' Priors: '''
        if p is None:
            p, log_F_nu, log_nu_a, log_nu_m= theta
        else:
            log_F_nu, log_nu_a, log_nu_m= theta

        if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 12 and 6 < log_nu_m < 12:
            return 0.0

        else:
            return -np.inf

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

    def get_labels(p=None):
        if p is None:
            return ['p','log F_v', 'log nu_a','log nu_m', 'log nu_c']
        else:
            return ['log F_v', 'log nu_a','log nu_m', 'log nu_c']

    # the model, must be named SED!!!
    def SED(nu, p, log_F_nu, log_nu_a, log_nu_m, log_nu_c, **kwargs):
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

    def lnprior(theta, nu, F, upperlimit, p=None, **kwargs):
        ''' Priors: '''
        uppertest = SyncfitModel._is_below_upperlimits(
            nu, F, upperlimit, theta, B4B5B3.SED, p=p
        )

        if p is None:
            p, log_F_nu, log_nu_a, log_nu_m, log_nu_c= theta
        else:
            log_F_nu, log_nu_a, log_nu_m, log_nu_c= theta
        if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 11 and log_nu_m < log_nu_a and log_nu_a < log_nu_c and uppertest:
            return 0.0

        else:
            return -np.inf

        def dynesty_transform(theta, nu, F, upperlimit, p=None, **kwargs):
            '''
            Prior transform for dynesty
            '''

            if p is None:
                p, log_F_nu, log_nu_a, log_nu_m, log_nu_c = theta
                fixed_p = False,
            else:
                fixed_p = True
                log_F_nu, log_nu_a, log_nu_m, log_nu_c = theta


            # log_F_nu between -4 and 2
            log_F_nu = log_F_nu*6 - 4

            # log_nu_a between 6 and 11
            log_nu_a = log_nu_a*5 + 6

            # same transform to log_nu_c
            log_nu_c = log_nu_c*5 + 6

            # same for log_nu_m
            log_nu_m = log_nu_m*5 + 6
            
            if not fixed_p:
                # p should be between 2 and 4
                p = 2*p + 2

                return p,log_F_nu,log_nu_a,log_nu_m,log_nu_c
            return log_F_nu,log_nu_a,log_nu_m,log_nu_c

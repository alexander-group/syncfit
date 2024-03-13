'''
Various models to use in MCMC fitting 
'''
import numpy as np

# Write some getters for things that are model specific
def get_pos(theta_init):
    ndim = len(theta_init)
    pos = np.array([np.array(theta_init) + [0.1*theta_init[0],
                                            0.1*theta_init[1]]*np.random.randn(ndim) for i in range(100)])
    return pos

def get_labels():
    return ['F_nu', 'nu_a']

def get_emcee_args(nu, F_muJy, F_error):
    p = 3
    nu = 1e9*nu
    F = np.array(F_muJy).astype(float)*1e-3
    F_error = np.array(F_error)*1e-3

    return {'p':p, 'nu':nu, 'F':F, 'F_error':F_error} 

# package those up for easy getting in do_emcee
def unpack_util(theta_init, nu, F_muJy, F_error):
    return get_pos(theta_init), get_labels(), get_emcee_args(nu, F_muJy, F_error)

# the model, must be named SED!!!
def SED(nu, p, F_nu, nu_a):
    b1 = 5/2
    b2 = (1-p)/2
    s = 1.25-0.18*p
    
    term = ((nu/nu_a)**(-s*b1)+(nu/nu_a)**(-s*b2))
    
    return F_nu*term**(-1/s)

# the other functions needed for emcee
def loglik(theta, nu, p, F, F_error):
    ''' Log Likelihood function '''
    F_nu, nu_a = theta
    model_result = SED(nu, p, F_nu, nu_a)
    sigma2 = F_error**2
    
    chi2 = np.sum((F - model_result)**2/sigma2)
    ll = -0.5*chi2
    return ll
        
def lnprior(theta):
    ''' Priors: '''
    F_nu, nu_a= theta
    if 1e-4 < F_nu < 50 and 1e6 < nu_a < 1e11:
        return 0.0
        
    else:
        return -np.inf    
        
def lnprob(theta, **kwargs):
    '''Keep or throw away step likelihood and priors'''
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + loglik(theta, **kwargs)

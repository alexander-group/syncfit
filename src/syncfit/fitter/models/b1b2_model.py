'''
Various models to use in MCMC fitting 
'''
import numpy as np

# Write some getters for things that are model specific
def get_pos(theta_init, nwalkers, p=None):
    ndim = len(theta_init)

    if p is None:
        pos = np.array([np.array(theta_init) + [0.1*theta_init[0],
    					        0.1*theta_init[1],
                                                0.1*theta_init[2],
                                                0.1*theta_init[3]]*np.random.randn(ndim) for i in range(nwalkers)])
    else:
        pos = np.array([np.array(theta_init) + [0.1*theta_init[0],
    					        0.1*theta_init[1],
                                                0.1*theta_init[2],
                                                ]*np.random.randn(ndim) for i in range(nwalkers)])
    
    return pos

def get_labels(p=None):
    if p is None:
        return ['p','log F_v', 'log nu_a','log nu_m']
    else:
        return ['log F_v', 'log nu_a','log nu_m']

def get_emcee_args(nu, F_muJy, F_error, p=None):
    nu = 1e9*nu
    F = np.array(F_muJy).astype(float)*1e-3
    F_error = np.array(F_error)*1e-3

    if p is None:
        return {'nu':nu, 'F':F, 'F_error':F_error} 
    else:
        return {'nu':nu, 'F':F, 'F_error':F_error, 'p':p} 
    
# package those up for easy getting in do_emcee
def unpack_util(theta_init, nu, F_muJy, F_error, nwalkers, p=None):
    return get_pos(theta_init,nwalkers,p), get_labels(p), get_emcee_args(nu, F_muJy, F_error, p)

# the model, must be named SED!!!
def SED(nu, p, log_F_nu, log_nu_a, log_nu_m):
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
    
    return F_nu * term1 * term2

# the other functions needed for emcee
def loglik(theta, nu, F, F_error, p=None, **kwargs):
    ''' Log Likelihood function '''
    if p is None:
        p, log_F_nu, log_nu_a, log_nu_m = theta
    else:
        log_F_nu, log_nu_a, log_nu_m = theta
        
    model_result = SED(nu, p, log_F_nu, log_nu_a, log_nu_m)
    sigma2 = F_error**2
    
    chi2 = np.sum((F - model_result)**2/sigma2)
    ll = -0.5*chi2
    return ll
        
def lnprior(theta, p=None, **kwargs):
    ''' Priors: '''
    if p is None:
        p, log_F_nu, log_nu_a, log_nu_m = theta
    else:
        log_F_nu, log_nu_a, log_nu_m = theta

    if 2< p < 4 and -4 < log_F_nu < 2 and 6 < log_nu_a < 12 and log_nu_m > log_nu_a:
        return 0.0
        
    else:
        return -np.inf    
        
def lnprob(theta, **kwargs):
    '''Keep or throw away step likelihood and priors'''
    lp = lnprior(theta, **kwargs)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + loglik(theta, **kwargs)
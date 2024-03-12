'''
Various models to use in MCMC fitting 
'''
pos = np.array([np.array(theta_init) + [0.1*theta_init[0],
                                            0.1*theta_init[1]]*np.random.randn(ndim) for i in range(100)])

def SED(nu, p, F_nu, nu_a):
    b1 = 5/2
    b2 = (1-p)/2
    s = 1.25-0.18*p
    
    term = ((nu/nu_a)**(-s*b1)+(nu/nu_a)**(-s*b2))
    
    return F_nu*term**(-1/s)

def loglik(theta, **kwargs):
    ''' Log Likelihood function '''
    model = SED(*theta, **kwargs)
    sigma2 = F_error**2
    
    chi2 = np.sum((F - model)**2/sigma2)
    ll = -0.5*chi2
    return ll
    
    
def lnprior(theta):
    ''' Priors: '''
    F_nu, nu_a= theta
    if 1e-4 < F_nu < 50 and 1e6 < nu_a < 1e11:
        return 0.0
        
    else:
        return -np.inf    
        
def lnprob(theta, nu, F, F_error, p):
    '''Keep or throw away step likelihood and priors'''
    lp = lnprior(theta, day, trans)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + loglik(theta, nu, F, F_error, p)

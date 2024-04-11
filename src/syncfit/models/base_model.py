'''
A BaseModel class that all the other models (including user custom models) are built
on. This allows for more flexibility and customization in the package.
'''
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    # Write some getters for things that are model specific
    # THESE WILL BE THE SAME ACROSS ALL MODELS!
    def get_pos(theta_init, nwalkers):
        ndim = len(theta_init)
        if not isinstance(theta_init, np.ndarray):
            theta_init = np.array(theta_init)

        diff = 0.01 # The amount to offset by, so 1 will give random pos between -1 and 1
        pos_offset = np.random.rand(nwalkers, ndim)*diff*2 - diff # *2 - 1 to put between -1 and 1 instead of 0 and 1
        pos = theta_init + pos_offset # this will offset the initial positions by pos_offset
        
        return pos
    
    def get_emcee_args(nu, F_muJy, F_error, p=None):
        nu = 1e9*nu
        F = np.array(F_muJy).astype(float)*1e-3
        F_error = np.array(F_error)*1e-3

        if p is None:
            return {'nu':nu, 'F':F, 'F_error':F_error} 
        else:
            return {'nu':nu, 'F':F, 'F_error':F_error, 'p':p} 

    # package those up for easy getting in do_emcee
    @classmethod
    def unpack_util(cls, theta_init, nu, F_muJy, F_error, nwalkers, p=None):
        return (cls.get_pos(theta_init,nwalkers),
                cls.get_labels(p=p),
                cls.get_emcee_args(nu, F_muJy, F_error, p))

    @classmethod
    def lnprob(cls, theta, **kwargs):
        '''Keep or throw away step likelihood and priors'''
        lp = cls.lnprior(theta, **kwargs)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + cls.loglik(theta, **kwargs)

    @classmethod
    def loglik(cls, theta, nu, F, F_error, p=None, **kwargs):
        ''' Log Likelihood function '''
        if p is not None:
            model_result = cls.SED(nu, p, *theta)
        else:
            model_result = cls.SED(nu, *theta)

        sigma2 = F_error**2
        
        chi2 = np.sum((F - model_result)**2/sigma2)
        ll = -0.5*chi2
        return ll
            
    # Some *required* abstract methods
    @staticmethod
    @abstractmethod
    def get_labels(*args, **kwargs):
        pass
        
    @staticmethod
    @abstractmethod
    def SED(*args, **kwargs):
        pass
    
    @staticmethod
    @abstractmethod
    def lnprior(*args, **kwargs):
        pass
    
    # override __subclasshook__
    @classmethod
    def __subclasshook__(cls, C):
        reqs = ['SED', 'lnprior', 'get_labels']
        if cls is BaseModel:
            if all(any(arg in B.__dict__ for B in C.__mro__) for arg in reqs):
                return True
        return NotImplemented

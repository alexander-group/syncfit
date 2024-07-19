'''
A BaseModel class that all the other models (including user custom models) are built
on. This allows for more flexibility and customization in the package.
'''
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import numpy as np

class _SyncfitModelMeta(type):
    '''
    This just gives all the subclasses for BaseModel the same docstrings
    for the inherited abstract methods
    '''
    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                member.__doc__ = getattr(bases[-1], name).__doc__
        return cls

class SyncfitModel(object, metaclass=_SyncfitModelMeta):
    '''
    An Abstract Base Class to define the basic methods that all syncfit
    models must contain. This will help maintain some level of standard for the models
    while also allowing users to customize their own.
    '''

    def __init__(self, prior, p=None):
        self.prior = prior
        self.labels = self.get_labels(p=p)
        self.ndim = len(self.labels)
        self.p = p
        
    # Write some getters for things that are model specific
    # THESE WILL BE THE SAME ACROSS ALL MODELS!
    @staticmethod
    def get_pos(theta_init:list, nwalkers:int) -> list[float]:
        '''
        Gets the initial position of all of the walkers assuming a gaussian distribution
        centered at theta_init.

        Args:
            theta_init (list): Initial location of the walkers
            nwalkers (int): Number of walkers

        Returns:
            A 2D array of the positions of all of the walkers
        '''
        ndim = len(theta_init)
        if not isinstance(theta_init, np.ndarray):
            theta_init = np.array(theta_init)

        diff = 0.01 # The amount to offset by, so 1 will give random pos between -1 and 1
        pos_offset = np.random.rand(nwalkers, ndim)*diff*2 - diff # *2 - 1 to put between -1 and 1 instead of 0 and 1
        pos = theta_init + pos_offset # this will offset the initial positions by pos_offset
        
        return pos

    @staticmethod
    def get_kwargs(nu:list, F_mJy:list, F_error:list, lum_dist:float=None,
                       t:float=None, upperlimit:list=None) -> dict:
        '''
        Packages up the args to be passed into the model based on the user input.

        Args:
            nu (list): frequencies in GHz
            F_mJy (list): Fluxes in milli janskies
            F_error (list): Flux errors in milli janskies
            p (float): A p-value to pass to the model, only used if p-value is fixed

        Returns:
            Dictionary of values, converted for the modeling used in the mcmc
        '''
        nu = 1e9*nu
        F = np.array(F_mJy).astype(float)
        F_error = np.array(F_error)

        base_args = {'nu':nu, 'F':F, 'F_error':F_error, 'upperlimit':upperlimit} 
        
        if lum_dist is not None:
            base_args['lum_dist'] = lum_dist

        if t is not None:
            base_args['t'] = t
            
        return base_args

    # package those up for easy getting in do_emcee
    def unpack_util(self, theta_init, nu, F_mJy, F_error, nwalkers, lum_dist=None,
                    t=None, upperlimit=None):
        '''
        A wrapper on the utility functions.

        Args:
            theta_init (list): List of initial theta locations
            nu (list): frequencies in GHz
            F_mJy (list): Fluxes in milli janskies
            F_error (list): Flux errors in milli janskies
            p (float): A p-value to pass to the model, only used if p-value is fixed
            nwalkers (int): THe number of walkers to use
        '''
        return (self.get_pos(theta_init,nwalkers),
                self.get_labels(p=self.p),
                self.get_kwargs(nu, F_mJy, F_error, upperlimit))

    def lnprob(self, theta:list, **kwargs):
        '''Keep or throw away step likelihood and priors

        Args:
            theta (list): location of the walker
            **kwargs: Any other arguments to be past to lnprior or loglik

        Returns:
            The likelihood of the data at that location
        '''
        lp = self.lnprior(theta, **kwargs)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.loglik(theta, **kwargs)

    def loglik(self, theta, nu, F, F_error, **kwargs):
        '''Log Likelihood function

        Args:
            theta (list): position of the walker
            nu (list): frequencies in GHz
            F_muJy (list): Fluxes in micro janskies
            F_error (list): Flux errors in micro janskies
            p (float): A p-value to pass to the model, only used if p-value is fixed

        Returns:
            The logarithmic likelihood of that theta position
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if self.p is not None:
                model_result = self.SED(nu, self.p, *theta, **kwargs)
            else:
                model_result = self.SED(nu, *theta, **kwargs)

        if not np.any(np.isfinite(model_result)):
            ll = -np.inf
        else:    
            sigma2 = F_error**2
        
            chi2 = np.sum((F - model_result)**2/sigma2)
            ll = -0.5*chi2
        
        return ll

    @staticmethod
    def _is_below_upperlimits(nu, F, upperlimits, theta, model):
        '''
        Checks that the location of theta is below any upperlimits
        '''

        if upperlimits is None:
            return True
        
        where_upperlimit = np.where(upperlimits)[0]
        F_upperlimits = F[where_upperlimit]

        if self.p is None:
            test_fluxes = model(nu, *theta)[where_upperlimit]
        else:
            test_fluxes = model(nu, self.p, *theta)[where_upperlimit]

        return np.all(F_upperlimits > test_fluxes)
    
    # Some *required* abstract methods
    def get_labels(self, *args, **kwargs):
        '''
        Describes a list of labels used in the return values of the mcmc chain.
        This varies depending on the inputs to the MCMC.
        '''
        return list(self.prior.keys())
        
    @abstractmethod
    def SED(self, *args, **kwargs):
        '''
        Describes the SED model for the model that subclasses this BaseModel
        '''
        pass

    def pack_theta(self, theta):
        '''
        Pack theta into a dictionary
        '''
        return {param:theta[idx] for idx, param in enumerate(self.labels)}
    
    def lnprior(self, theta, nu, F, upperlimit, **kwargs):
        '''
        Logarithmic prior function that can be changed based on the SED model.
        '''
        uppertest = SyncfitModel._is_below_upperlimits(
            nu, F, upperlimit, theta, self.SED
        )

        packed_theta = self.pack_theta(theta)
        
        all_res = []
        for param, val in self.prior.items():
            res = val[0] < packed_theta[param] < val[1]
            all_res.append(res)

        if all(all_res) and uppertest:
            return 0.0
        else:
            return -np.inf

    def _transform_dynesty_transform(self, param, val):
        '''
        Subfunction for transforming the dynesty inputs to the correct prior range
        '''
        diff = abs(val[0]-val[1])
        if val[0] == val[1] or (val[0] == 0 and val[1] == 0) or val[0] > val[1]:
            raise ValueError(f'Invalid prior range for {param}!')

        elif val[0] == 0 and val[1] != 0:
            return param*val[1]

        elif val[0] != 0 and val[1] == 0:
            return param*val[0]

        elif val[0] > 0 and val[1] > 0:
            linear_shift = min(val)
            return param*diff + linear_shift

        elif val[0] < 0 and val[1] > 0:
            linear_shift = val[1]
            return param*diff - linear_shift

        elif val[0] < 0 and val[0] < 0:
            linear_shift = abs(max(val))
            return -param*diff - linear_shift

        else:
            raise ValueError('This is a prior range we had not considered!')

    def dynesty_transform(self, theta, **kwargs):
        '''
        Transform the input values to the correct prior ranges for dynesty
        '''
        packed_theta = self.pack_theta(theta)
        return tuple(
            self._transform_dynesty_transform(
                packed_theta[p],v
            ) for p,v in self.prior.items()
        )
            
    # override __subclasshook__
    @classmethod
    def __subclasshook__(cls, C):
        reqs = ['SED', 'lnprior', 'get_labels']
        if cls is BaseModel:
            if all(any(arg in B.__dict__ for B in C.__mro__) for arg in reqs):
                return True
        return NotImplemented

    # add a register method so users don't have to create a new class
    @classmethod
    def override(cls,func):
        '''
        This method should be used as a decorator to override other methods 
        '''
        exec(f'cls.{func.__name__} = func')

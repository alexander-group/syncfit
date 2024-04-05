'''
Code to run the MCMC using the models in models.py
'''
import importlib
import numpy as np
import matplotlib.pyplot as plt
import emcee
from .analysis import *
from .models.b5_model import B5
from .models.base_model import BaseModel

def do_emcee(theta_init, nu, F_muJy, F_error, model=B5, niter=2000,
             nwalkers=100, fix_p=None, day=None, plot=False):

    """
    nu - frequency in GHz
    F_muJy - flux in micro janskies
    F_error - flux error in micro janskies
    fix_p: Will fix the p value to whatever you give, do not provide p in theta_init
             if this is the case!
    day: day of observation (string)
    theta_init: array of initial guess 
    model: Model class to use from syncfit.fitter.models, must be a subclass of BaseModel!

    Returns:
    flat_samples, log_prob
    """
    
    # Check that the model subclasses BaseModel
    # if issubclass(model, BaseModel):
    #    raise ValueError('Input model is not a subclass of BaseModel!!')
    
    ### Fill in initial guesses and number of parameters  
    theta_init = np.array(theta_init)
    ndim = len(theta_init)

    # get some values from the import
    pos, labels, emcee_args = model.unpack_util(theta_init, nu, F_muJy, F_error,
                                                nwalkers, p=fix_p)
    
    # setup and run the MCMC
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.lnprob, kwargs=emcee_args)
    pos, prob, state = sampler.run_mcmc(pos, niter, progress=True);

    flat_samples, log_prob = extract_output(sampler, discard=niter//2)
    
    if plot:
        
        # plot the chains
        fig, ax = plot_chains(sampler, labels)
        
        #Print best fit parameters
        print('Best fit values for day: ', day)
        get_bounds(sampler, labels, toprint=True)
        
        # get the best 100 of the chain (ie where log_prob is maximum)
        # then plot these
        if 'p' in emcee_args:
            fig, ax = plot_best_fit(model, sampler, emcee_args['nu'], emcee_args['F'],
                                    p=emcee_args['p'])
        else:
            fig, ax = plot_best_fit(model, sampler, emcee_args['nu'], emcee_args['F'])
        
    return sampler

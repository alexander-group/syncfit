'''
Code to run the MCMC using the models in models.py
'''
from pickle import PicklingError
import importlib
import numpy as np
import matplotlib.pyplot as plt
import emcee
import dynesty
from multiprocessing import Pool
from .analysis import *
from .models.mq_model import MQModel
from .models.syncfit_model import SyncfitModel

def do_dynesty(nu:list[float], F_mJy:list[float], F_error:list[float],
               lum_dist:float=None, t:float=None,
               model:SyncfitModel=MQModel, fix_p:float=None,
               upperlimits:list[bool]=None, ncores:int=1, seed:int=None,
               **dynesty_kwargs
             ) -> tuple[list[float],list[float]]:
    """
    Fit the data with the given model using the dynesty nested sampling package

    Args:
        nu (list): list of frequencies in GHz
        F_muJy (list): list of fluxes in milli janskies
        F_error (list): list of flux error in milli janskies
        model (SyncfitModel): Model class to use from syncfit.fitter.models. Can also be a custom model
                           but it must be a subclass of SyncfitModel!
        lum_dist (float): luminosity distance in cgs units. Only needed for MQModel. Default is None.
        t (flost): observation time in seconds. Only needed for MQModel. Default is None.
        fix_p (float): Will fix the p value to whatever you give, do not provide p in theta_init
                               if this is the case!
        upperlimits (list[bool]): True if the point is an upperlimit, False otherwise.
        ncores (int) : The number of cores to run on, default is 1 and won't multiprocess
        seed (int): The seed for the random number generator passed to dynesty,
        dynesty_kwargs : kwargs to pass to dynesty 
    Returns:
        flat_samples, log_prob
    """
    if isinstance(model(), MQModel) and (lum_dist is None or t is None):
        raise ValueError('lum_dist and t reequired for MQModel!')
    
    # get the extra args
    dynesty_args = model.get_kwargs(nu, F_mJy, F_error, lum_dist=lum_dist, t=t, p=fix_p)
    ndim = len(model.get_labels(p=fix_p))
    rstate = np.random.default_rng(seed)
    
    # construct the sampler and run it
    # NOTE: I give it the lnprob instead of loglik because there can be some other
    # priors that are built into the lnprob that can not be in the dynesty prior
    # transformation function (for example, the upperlimit testing).
    # if the transormation is done correctly and there are no other priors (like no
    # upperlimits) then lnprob is identical to loglik so this is okay to do!
    with Pool(ncores) as pool:
        pool.size = ncores
        dsampler = dynesty.DynamicNestedSampler(model.lnprob, model.dynesty_transform,
                                                ndim=ndim, rstate=rstate,
                                                logl_kwargs=dynesty_args,
                                                ptform_kwargs=dynesty_args,
                                                pool=pool, **dynesty_kwargs)
        try:
            dsampler.run_nested()
        except PicklingError:
            raise ValueError(
                'The override decorator syntax is not currently supported for dynesty!'
            )
            
    return dsampler        

def do_emcee(theta_init:list[float], nu:list[float], F_mJy:list[float],
             F_error:list[float], lum_dist:float=None, t:float=None,
             model:SyncfitModel=SyncfitModel, niter:int=2000,
             nwalkers:int=100, fix_p:float=None, upperlimits:list[bool]=None,
             day:str=None, plot:bool=False, ncores:int=1
             ) -> tuple[list[float],list[float]]:
    """
    Fit the data with the given model using the emcee package.
    
    Args:
        theta_init (list): array of initial guesses, must be the length expected by model
        nu (list): list of frequencies in GHz
        F_mJy (list): list of fluxes in milli janskies
        F_error (list): list of flux error in milli janskies
        model (SyncfitModel): Model class to use from syncfit.fitter.models. Can also be a custom model
                           but it must be a subclass of SyncfitModel!
        niter (int): The number of iterations to run on.
        nwalkers (int): The number of walkers to use for emcee
        fix_p (float): Will fix the p value to whatever you give, do not provide p in theta_init
                               if this is the case!
        lum_dist (float): luminosity distance in cgs units. Only needed for MQModel. Default is None.
        t (flost): observation time in seconds. Only needed for MQModel. Default is None.
        upperlimits (list[bool]): True if the point is an upperlimit, False otherwise.
        day (string): day of observation, used for labeling plots
        plot (bool): If True, generate the plots used for debugging. Default is False.
        ncores (int) : The number of cores to run on, default is 1 and won't multiprocess
    Returns:
        flat_samples, log_prob
    """
    if isinstance(model(), MQModel) and (lum_dist is None or t is None):
        raise ValueError('lum_dist and t reequired for MQModel!')
    
    ### Fill in initial guesses and number of parameters  
    theta_init = np.array(theta_init)
    ndim = len(theta_init)

    # get some values from the import
    nu = np.array(nu)
    F_mJy = np.array(F_mJy)
    F_error = np.array(F_error)
    if upperlimits is not None:
        upperlimits = np.array(upperlimits)
        
    pos, labels, emcee_args = model.unpack_util(theta_init, nu, F_mJy, F_error,
                                                nwalkers, p=fix_p, lum_dist=lum_dist,
                                                t=t, upperlimit=upperlimits)
    
    # setup and run the MCMC
    nwalkers, ndim = pos.shape

    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            model.lnprob,
            pool=pool,
            kwargs=emcee_args
        )
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

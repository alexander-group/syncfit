'''
Some useful plotting code from the outputs of the mcmc
'''

import matplotlib.pyplot as plt
from .util import *
from ..models import MQModel

def plot_chains(sampler, labels, fig=None, axes=None):
    '''
    Plot the flattened chains

    Args:
        sampler [emcee.EnsembleSampler]: Emcee EnsembleSampler object after running the MCMC
        labels [list[str]]: list of labels corresponding to the chains
        fig [matplotlib.pyplot.Figure]: Matplotlib Figure object, Default is None and one will be created.
        axes [matplotlib.pyplot.Axis]: list of Matplotlib Axis object, Default is None and one will be created. 

    Returns:
        matplotlib fig, ax
    '''

    samples = sampler.get_chain()
    ndim = samples.shape[-1]
    
    if axes is None:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    assert len(axes) == ndim, "Length of axes must be the same as the number of dimensions in the sampler"
        
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");

    return fig, axes

def plot_best_fit(model, sampler, nu, F, Ferr, nkeep=1000, lum_dist=None,
                  nu_arr=None, method='random', fig=None, ax=None, day=None):
    '''
    Plot best fit model

    Args:
        model [syncfit.fitter.models]: A module from syncfit.fitter.models used to fit the data
        sampler [emcee.EnsembleSampler]: Emcee EnsembleSampler object after running the MCMC
        nu [list]: the observed frequencies
        F [list]: The observed flux densities
        Ferr [list]: The observed flux error
        nkeep [int]: Number of values to keep
        nu_arr [list]: List of nus for the best fit lines to be plot with
        method [str]: Either 'max' or 'last' or 'random', default is max.
                      - max: takes the nkeep maximum probability values
                      - last: takes the last nkeep values from the chain
                      - random: Chooses the nkeep values from the last nkeep*10 values in the chain
        fig [matplotlib.pyplot.Figure]: Matplotlib Figure object, Default is None and one will be created.
        axes [matplotlib.pyplot.Axis]: list of Matplotlib Axis object, Default is None and one will be created. 

    Returns:
        matplotlib fig, ax
    '''
    if isinstance(model, MQModel) and model.t is not None:
        t = model.t
    
    flat_samples, log_prob = extract_output(sampler)
    
    if method == 'max':
        toplot = flat_samples[np.argsort(log_prob)[-nkeep:]]
    elif method == 'last':
        toplot = flat_samples[-nkeep:]
    elif method == 'random':
        toplot = flat_samples[-nkeep*10:][np.random.randint(0, nkeep*10, nkeep)] 
    else:
        raise ValueError('method must be either last or max!')

    if nu_arr is None:
        nu_plot = np.arange(1e8,3e11,1e7)
    else:
        nu_plot = nu_arr
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    if lum_dist is not None and model.t is None:
        kwargs = dict(lum_dist=lum_dist)
    elif lum_dist is not None and model.t is not None:
        kwargs = dict(lum_dist=lum_dist, t=t)
    else:
        kwargs={}
        
    for val in toplot:
        if model.p is not None:
            res = model.SED(nu_plot, model.p, *val, **kwargs)
        else:
            res = model.SED(nu_plot, *val, **kwargs)
            
        ax.plot(nu_plot, res,
                '-', color='cornflowerblue', lw = 0.5, alpha = 0.1)
        
    ax.errorbar(nu, F, yerr=Ferr, fmt='o', markerfacecolor='none', markeredgecolor='k')

    if day is not None:
        ax.text(1.25e9,2e-2,s='Day '+ day, fontsize = 20)

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

    return fig, ax

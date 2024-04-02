'''
Some useful plotting code from the outputs of the mcmc
'''

import matplotlib.pyplot as plt
from .util import *

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

def plot_best_fit(model, sampler, nu, F, nkeep=1000, p=None, method='max', fig=None, ax=None, day=None):
    '''
    Plot best fit model

    Args:
        model [syncfit.fitter.models]: A module from syncfit.fitter.models used to fit the data
        sampler [emcee.EnsembleSampler]: Emcee EnsembleSampler object after running the MCMC
        nu [list]: the observed frequencies
        F [list]: The observed flux densities
        nkeep [int]: Number of values to keep
        p [float]: p-value used, if not None
        method [str]: Either 'max' or 'last', default is max.
        fig [matplotlib.pyplot.Figure]: Matplotlib Figure object, Default is None and one will be created.
        axes [matplotlib.pyplot.Axis]: list of Matplotlib Axis object, Default is None and one will be created. 

    Returns:
        matplotlib fig, ax
    '''

    flat_samples, log_prob = extract_output(sampler)
    
    if method == 'max':
        toplot = flat_samples[np.argsort(log_prob)[-nkeep:]]
    elif method == 'last':
        toplot = flat_samples[-nkeep:]
        
    nu_plot = np.arange(1e8,3e11,1e7)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    
    for val in toplot:
        if p is not None:
            res = model.SED(nu_plot, p, *val)
        else:
            res = model.SED(nu_plot, *val)
            
        ax.plot(nu_plot, res,
                '-', color='k', lw = 0.5, alpha = 0.1)
        
    ax.errorbar(nu, F, fmt='o', markeredgecolor='black',
                markeredgewidth=3, markersize=15, c='r')

    if day is not None:
        ax.text(1.25e9,2e-2,s='Day '+ day, fontsize = 20)
    #ax.set_ylim(8e-4,55)
    #ax.set_xlim(0.1e9,2e11)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

    return fig, ax

'''
Code to run the MCMC using the models in models.py
'''
import importlib
import numpy as np
import matplotlib.pyplot as plt
import emcee

def do_emcee(theta_init, nu, F_muJy, F_error, toimport='single_break_model', niter=2000, day=None, plot=False):

    """
    nu - frequency in GHz
    F_muJy - flux in micro janskies
    F_error - flux error in micro janskies
    day: day of observation (string)
    theta_init: array of initial guess 
    """
        
    # import the relevant module
    model = importlib.import_module(f'syncfit.{toimport}')

    ### Fill in initial guesses and number of parameters  
    theta_init = np.array(theta_init)
    ndim = len(theta_init)

    # get some values from the import
    pos, labels, emcee_args = model.unpack_util(theta_init, nu, F_muJy, F_error)
    
    # setup and run the MCMC
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.lnprob, kwargs=emcee_args)
    pos, prob, state = sampler.run_mcmc(pos, niter, progress=True);

    flat_samples = sampler.get_chain(discard=niter//2, flat=True)

    if plot:
        #Plot chains
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");

        #Print best fit parameters
        print('Best fit values for day: ', day)
        p_value = []
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            vals = [mcmc[1],(q[0]+q[1])/2]
            txt = "\mathrm{{{3}}} = {0:.2e}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            print(txt)

        nu_plot = np.arange(1e8,3e11,1e7)
        fig = plt.figure(figsize = (4,4))
        ax = plt.subplot(111)
        v = len(flat_samples.transpose()) - 1000 # only plot the last thousange
        for i in range(len(flat_samples.transpose()[0][v:])):
            ax.plot(nu_plot, model.SED(nu_plot, emcee_args['p'],
                                       *[val[v:][i] for val in flat_samples.transpose()]
                                       ),
                    '-', color='k', lw = 0.5, alpha = 0.1)

        ax.errorbar(emcee_args['nu'], emcee_args['F'], fmt='o', markeredgecolor='black',
                    markeredgewidth=3, markersize=15, c='r')
        if day is not None:
            ax.text(1.25e9,2e-2,s='Day '+ day, fontsize = 20)
        #ax.set_ylim(8e-4,55)
        #ax.set_xlim(0.1e9,2e11)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()
    
    return flat_samples

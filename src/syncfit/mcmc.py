'''
Code to run the MCMC using the models in models.py
'''
import importlib
import numpy as np
import matplotlib.pyplot as plt
import emcee

def do_emcee(theta_init, nu, F_muJy, F_error, model_name='b5', niter=2000,
             nwalkers=100, fix_p=None, day=None, plot=False):

    """
    nu - frequency in GHz
    F_muJy - flux in micro janskies
    F_error - flux error in micro janskies
    fix_p: Will fix the p value to whatever you give, do not provide p in theta_init
             if this is the case!
    day: day of observation (string)
    theta_init: array of initial guess 
    model_name: Shortened model name to use. Options are b5, b4b5, b4b5b3, b1b2, b1b2_b3b4_weighted

    Returns:
    flat_samples, log_prob
    """
        
    # import the relevant module
    model = importlib.import_module(f'syncfit.{model_name}_model')

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

    flat_samples = sampler.get_chain(discard=niter//2, flat=True)
    log_prob = sampler.get_log_prob(discard=niter//2, flat=True)
    
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
            
        # get the best 100 of the chain (ie where log_prob is maximum)
        # then plot these
        nkeep = 1000
        #import pdb; pdb.set_trace()
        toplot = flat_samples[np.argsort(log_prob)[-nkeep:]]
        
        nu_plot = np.arange(1e8,3e11,1e7)
        fig = plt.figure(figsize = (4,4))
        ax = plt.subplot(111)
        for val in toplot:
            if 'p' in emcee_args:
                res = model.SED(nu_plot, emcee_args['p'], *val)
            else:
                res = model.SED(nu_plot, *val)
                
            ax.plot(nu_plot, res,
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
    
    return flat_samples, log_prob

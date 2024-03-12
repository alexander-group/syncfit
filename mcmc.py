'''
Code to run the MCMC using the models in models.py
'''

def do_emcee(df, theta_init, emcee_args, labels, toimport='single_break_model', niter=2000, day='Unknown', plot=False):

    """
    df: Dataframe with data 
    	nu - frequency in GHz
    	F_muJy - flux in micro janskies
    	F_error - flux error in micro janskies
    day: day of observation (string)
    theta_init: array of initial guess 
    """    
    # import the relevant module
    exec(f'from {toimport} import *')

    p = 3
    nu = 1e9*df.nu
    F = np.array(df.F_muJy).astype(float)*1e-3
    F_error = np.array(df.F_error)*1e-3

    
    ### Fill in initial guesses and number of parameters  
    theta_init = np.array(theta_init)
    ndim = len(theta_init)

    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=*emcee_args)
    pos, prob, state = sampler.run_mcmc(pos, niter, progress=True);

    flat_samples = sampler.get_chain(discard=1000, flat=True)

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
            display(Math(txt))

        nu_plot = np.arange(1e8,3e11,1e7)
        fig = plt.figure(figsize = (4,4))
        ax = plt.subplot(111)
        for i in range(len(flat_samples.transpose()[0][99000:])):
            ax.plot(nu_plot, SED(nu_plot, p, flat_samples.transpose()[0][99000:][i], flat_samples.transpose()[1][99000:][i]), '-', color='k', lw = 0.5, alpha = 0.1)

        ax.errorbar(nu,F, fmt = 'o', markeredgecolor = 'black', markeredgewidth=3, markersize= 15, c = 'r')
        ax.text(1.25e9,2e-2,s='Day '+ day, fontsize = 20)
        ax.set_ylim(8e-4,55)
        ax.set_xlim(0.1e9,2e11)
        ax.set_yscale('log')
        ax.set_xscale('log')
        aesthetic(ax)
        plt.show()
    
    return flat_samples

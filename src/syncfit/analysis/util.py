'''
Some useful analysis functions
'''

import numpy as np

def extract_output(sampler, **kwargs):
    '''
    Extracts the flattened samples and log_prob from sampler
    '''

    flat_samples = sampler.get_chain(flat=True, **kwargs)
    log_prob = sampler.get_log_prob(flat=True, **kwargs)

    return flat_samples, log_prob

def get_ndim(sampler):
    '''
    Gets the number of dimensions from an Emcee sampler
    '''
    flat_samples, _ = extract_output(sampler)
    return len(flat_samples[0])

def get_bounds(sampler, labels, toprint=False, **kwargs):
    '''
    Extract the median, upper, and lower limits from the flat_samples chain
    '''
    
    flat_samples, _ = extract_output(sampler, **kwargs)
    ndim = get_ndim(sampler)
    
    medians, upper, lower = [], [], []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        vals = [mcmc[1],(q[0]+q[1])/2]
        if toprint:
            txt = "\mathrm{{{3}}} = {0:.2e}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[i])
            print(txt)

        medians.append(mcmc[1])
        upper.append(q[1])
        lower.append(q[0])

    return medians, upper, lower

        

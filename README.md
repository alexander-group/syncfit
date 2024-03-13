# syncfit
Modeling code for Radio Synchrotron SEDs that uses MCMC and the SED models from XXX and YYY (fill in citations later).

# Installation
Run the following commands in a terminal (assuming git is installed)
```
git clone https://github.com/alexander-group/syncfit.git
cd syncfit
pip install -e .
```

# Running the code
A simple example if you have arrays of flux, flux error, and a central frequency in GHz
```
theta_init = [0,0]
chain = syncphot.mcmc.do_emcee(theta_init, nu, F, Ferr)
```

# Development instructions
To add a new model to the code create a new *descriptive* python file in `src/syncphot`. See `single_break_model.py` for an example but this model file must have the following functions
* SED: The SED model
* lnprob
* loglik
* lnprob
* get_pos: get the initial pos array
* get_labels: labels for plotting
* get_emcee_args: gets any extra constant arguments besides theta that are passed into the SED model
* unpack_util: see `single_break_model.py` for how to write this method
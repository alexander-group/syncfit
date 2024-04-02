'''
Code to convert the outputs from the synchrotron fitting to physics quantities

These models are from Barniol Duran 2013
'''

def Req(F_p_mJy:float, d_L:float, nu_a:float, nu_p:float, z:float, t_d:float,
        f_A:float=1, f_V:float=1) -> float:
    '''
    Computes the equipartition radius from observable quantities.

    Eq. (4) in Barniol Duran (2013)

    Args:
        F_p_mJy [float]: The peak flux density in mJy
        d_L [float]: The luminosity distance in cm
        nu_a [float]: The self-absorption break frequency in Hz
        nu_p [float]: The frequency associated with the peak flux in Hz
        z [float]: The redshift
        t_d [float]: The time since in the onset of the relativistic outflow in days
        f_A [float]: The area geometric factor, default is 1
        f_V [float]: The volume geometric factor, default is 1

    Returns:
        The equipartition radius in cm
    '''

    # convert to the units used in the paper
    d_L_28, nu_p_10, eta = _mini_converter(d_L, nu_p, nu_a)

    prefactor = 7.5e17 #cm
    factor = F_p_mJy**(2/3) * d_L_28**(4/3) * nu_p_10**(-17/12) * eta**(35/36) * (1+z)**(-5/3) * t_d**(-5/12)
    postfactor = f_A**(-7/12) * f_V**(-1/12)

    return prefactor * factor * postfactor

def lagrange_factor(F_p_mJy:float, d_L:float, nu_a:float, nu_p:float, z:float, t_d:float,
        f_A:float=1, f_V:float=1) -> float:
    '''
    Computes the lagrange factor (\Gamma) from observable quantities.

    Eq. (5) in Barniol Duran (2013)

    Args:
        F_p_mJy [float]: The peak flux density in mJy
        d_L [float]: The luminosity distance in cm
        nu_a [float]: The self-absorption break frequency in Hz
        nu_p [float]: The frequency associated with the peak flux in Hz
        z [float]: The redshift
        t_d [float]: The time since in the onset of the relativistic outflow in days
        f_A [float]: The area geometric factor, default is 1
        f_V [float]: The volume geometric factor, default is 1

    Returns:
        The Lagrange Factor (unitless)
    '''

    # convert to the units used in the paper
    d_L_28, nu_p_10, eta = _mini_converter(d_L, nu_p, nu_a)

    prefactor = 12 #cm
    factor = F_p_mJy**(1/3) * d_L_28**(2/3) * nu_p_10**(-17/24) * eta**(35/72) * (1+z)**(-1/3) * t_d**(-17/24)
    postfactor = f_A**(-7/24) * f_V**(-1/24)

    return prefactor * factor * postfactor

def Eeq(F_p_mJy:float, d_L:float, nu_a:float, nu_p:float, z:float, t_d:float,
        f_A:float=1, f_V:float=1) -> float:
    '''
    Computes the equipartition energy from observable quantities.

    Eq. (6) in Barniol Duran (2013)

    Args:
        F_p_mJy [float]: The peak flux density in mJy
        d_L [float]: The luminosity distance in cm
        nu_a [float]: The self-absorption break frequency in Hz
        nu_p [float]: The frequency associated with the peak flux in Hz
        z [float]: The redshift
        t_d [float]: The time since in the onset of the relativistic outflow in days
        f_A [float]: The area geometric factor, default is 1
        f_V [float]: The volume geometric factor, default is 1

    Returns:
        The equipartition energy in ergs
    '''

    # convert to the units used in the paper
    d_L_28, nu_p_10, eta = _mini_converter(d_L, nu_p, nu_a)

    prefactor = 5.7e47 # erg
    factor = F_p_mJy**(2/3) * d_L_28**(4/3) * nu_p_10**(1/12) * eta**(5/36) * (1+z)**(-5/3) * t_d**(13/12)
    postfactor = f_A**(-1/12) * f_V**(5/12)

    return prefactor * factor * postfactor

def _mini_converter(d_L, nu_p, nu_a):
    '''
    Converts to the appropriate units for the paper
    '''
    return d_L/1e28, nu_p/1e10, nu_p/nu_a

import logging
import numpy as np

from .utils import *

def GbandCaI(x_in: float, y_in: float) -> float:
    """
    Calculates the Gband and Ca I index used in Rough_type_2.

    Parameters
    ----------
    x_in : float
        Input value for Ca I.
        
    y_in : float
        Input value for Gband.

    Returns
    -------
    index : float
        The calculated Gband and Ca I index.
    """
    # Coefficients
    a = -130.53346170487083
    b = 303.48986510742679
    c = 323.94730249323806
    d = -316.88518509163600
    f = -181.21782978289525
    g = -144.38268090700953
    
    # Calculate the index
    index = (a +
             b * x_in + 
             c * y_in +
             d * x_in**2 + 
             f * y_in**2 + 
             g * x_in * y_in)
    
    return index

def CaKHe4471(x_in: float, y_in: float) -> float:
    """
    Calculates the CaK and He 4471 index using a polynomial function.
    
    Parameters
    ----------
    x_in : float
        Input value for the CaK index.
        
    y_in : float
        Input value for the He 4471 index.

    Returns
    -------
    index : float
        The calculated index value.
    """
    # Coefficients
    a = -161.77868547180907
    b = 206.5735163610168
    c = 139.34317702166081
    d = -38.066771401657036
    f = 148.60705532285428
    g = -238.6798600951094
    
    # Calculate the polynomial value
    index = (a +
            b * x_in +
            c * y_in +
            d * x_in**2 +
            f * y_in**2 +
            g * x_in * y_in)
    
    return index

def TiOIndex(x: float) -> float:
    """
    Calculates the TiO index based on the given value using a linear equation. The value is capped at 45.5 if it exceeds this limit.

    Parameters
    ----------
    x : float
        The input value for which the TiO index is calculated.

    Returns
    -------
    index : float
        The TiO index, capped at a maximum value of 45.5.
    """
    index = 26.8899 + 16.8833 * x
    return min(index, 45.5)

def Carbon(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Calculates the Carbon index based on specific wavelength ranges.

    Parameters
    ----------
    wave : np.ndarray
        Array of wavelength values.
        
    flux : np.ndarray
        Array of flux values.

    Returns
    -------
    index : float
        The Carbon index.
    """

    # Define wavelength ranges
    mask1 = (wave >= 4675.0) & (wave <= 4725.0)
    mask2 = (wave >= 4755.0) & (wave <= 4805.0)
    mask3 = (wave >= 5070.0) & (wave <= 5140.0)
    mask4 = (wave >= 5176.0) & (wave <= 5246.0)

    # Sum fluxes within the specified wavelength ranges
    sum1 = np.sum(flux[mask1])
    sum2 = np.sum(flux[mask2])
    sum3 = np.sum(flux[mask3])
    sum4 = np.sum(flux[mask4])

    # Calculate and return the Carbon index
    return (sum2 / sum1) + (sum4 / sum3)

def hyd_rat(wave: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    """
    Computes the width and ratio of the Hydrogen gamma line.
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength array.
        
    flux : np.ndarray
        Flux array.
    
    Returns
    -------
    width : float
        Width of the Hydrogen gamma line.
        
    ratio : float
        Ratio of the width to the depth of the Hydrogen gamma line.
    """
    # Calculate sums and find the minimum flux in the specified range #
    sum1 = np.mean(flux[(wave >= 4190.0) & (wave <= 4230.0)])
    sum2 = np.mean(flux[(wave >= 4450.0) & (wave <= 4500.0)])
    bot = np.min(flux[(wave >= 4330.0) & (wave <= 4350.0)])
    
    slope = (sum2 - sum1) / 265.0
    b = sum1 - slope * 4210.0
    height = slope * 4340.0 + b
    depth = height - bot
    depth2 = depth / 2
    depth /= height
    
    logging.info(f"NN: slope = {slope:.6f}, height = {height:.6f}, depth = {depth:.6f}")
    
    # Find the index closest to the center of the H gamma line (4340.4) #
    j = np.argmin(np.abs(wave - 4340.4))
    
    # Find blue wing midpoint #
    k = j
    while slope * wave[k] + (b - depth2) - flux[k] >= 0:
        k -= 1
    wid1 = wave[k]
    
    # Find red wing midpoint #
    k = j
    while slope * wave[k] + (b - depth2) - flux[k] >= 0:
        k += 1
    wid2 = wave[k]
    
    logging.info(f"wid1 = {wid1:.6f}, wid2 = {wid2:.6f}")
    
    # Calculate width and ratio
    width = wid2 - wid1
    ratio = width / (depth if depth != 0.0 else 1)
    
    return width, ratio

def late_M(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Calculates the ratio of flux sums for identifying late M-type stars.
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength array.
        
    flux : np.ndarray
        Flux array.
    
    Returns
    -------
    ratio : float
        Ratio of the flux sums in the specified wavelength ranges.
    """
    # Calculate sums in the specified wavelength ranges #
    sum1 = np.sum(flux[(wave >= 4918.0) & (wave <= 4948.0)])
    sum2 = np.sum(flux[(wave >= 4958.0) & (wave <= 4988.0)])
    
    # Return the ratio of the sums
    return sum1 / sum2

def DB(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Computes the ratio for identifying DB-type stars.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array.
        
    flux : np.ndarray
        Flux array.

    Returns
    -------
    ratio : float
        The ratio of continuum to line flux for the specified wavelength ranges.
    """
    # Calculate continuum and line sums using logical indexing #
    cont1 = np.sum(flux[(wave >= 4301.0) & (wave <= 4351.0)])
    cont2 = np.sum(flux[(wave >= 4591.0) & (wave <= 4641.0)])
    sum_line = np.sum(flux[(wave >= 4421.0) & (wave <= 4521.0)])

    # Calculate the continuum and line values #
    cont = (cont1 + cont2) / 100.0
    line = sum_line / 100.0

    # Calculate the ratio #
    ratio = cont / line

    return ratio

def DZ(wave: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    """
    Computes the DZ star ratio and CN band ratio (C_ratio) for distinguishing DZ stars from carbon-enhanced stars.
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength array.
        
    flux : np.ndarray
        Flux array.
    
    Returns
    -------
    ratio : float
        Ratio used to identify DZ stars.
        
    C_ratio : float
        Ratio used to distinguish DZ stars from carbon-enhanced stars.
    """
    # Calculate the continuum and line sums #
    cont = np.sum(flux[(wave > 3850.0) & (wave <= 3900.0)]) + np.sum(flux[(wave > 4000.0) & (wave <= 4050.0)])
    line = np.sum(flux[(wave > 3925.0) & (wave <= 3940.0)]) + np.sum(flux[(wave > 3960.0) & (wave <= 3975.0)])
    
    # Calculate the DZ ratio
    ratio = (30.0 * cont) / (line * 100) if line != 0 else -9.99
    
    # Calculate the sums for CN band check #
    sum1 = np.sum(flux[(wave >= 4043.0) & (wave <= 4088.0)]) + np.sum(flux[(wave >= 4219.0) & (wave <= 4264.0)])
    sum2 = np.sum(flux[(wave >= 4140.0) & (wave <= 4210.0)])
    
    # Calculate C_ratio for distinguishing DZ from carbon-enhanced stars
    C_ratio = sum1 / sum2 if sum2 != 0 else -9.99
    
    return ratio, C_ratio

def DO(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Computes the ratio used to identify DO stars by comparing continuum and line fluxes.
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength array.
        
    flux : np.ndarray
        Flux array.
    
    Returns
    -------
    ratio : float
        The ratio of continuum to line fluxes, used to identify DO stars.
    """
    # Calculate the continuum and line sums #
    cont = np.sum(flux[(wave >= 4780.0) & (wave <= 4820.0)]) + np.sum(flux[(wave >= 4900.0) & (wave <= 4940.0)])
    line = np.sum(flux[(wave >= 4820.0) & (wave <= 4900.0)])
    
    # Calculate the ratio
    ratio = cont / line if line != 0 else -9.99
    
    return ratio

def hyd_D2(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Determines the width of the H gamma line at mid-depth to distinguish between a normal star and a DA white dwarf.

    Parameters
    ----------
    wave : numpy.ndarray
        Array of wavelength values.
    
    flux : numpy.ndarray
        Array of flux values.

    Returns
    -------
    D2 : float
        The width of the H gamma line at mid-depth, or 0.0 if the measurement is not valid.
    """
    # Compute sums and continuum values and find minimum flux in specified wavelength ranges
    mask = (wave >= 4200.) & (wave <= 4220.)
    sum1, cont1 = np.sum(flux[mask]), np.mean(flux[mask])
    
    mask = (wave >= 4510.) & (wave <= 4570.)
    sum2, cont2 = np.sum(flux[mask]), np.mean(flux[mask])
    
    mask = (wave > 4330.) & (wave < 4350.)
    min_index = np.argmin(flux[mask])
    wave_min, flux_min = wave[mask][min_index], flux[mask][min_index]

    logging.info(f"sum1 = {sum1:.6f} sum2 = {sum2:.6f} wavemin = {wave_min:.6f}")

    # Compute D2 value
    cont = cont1 + (cont2 - cont1) * (wave_min - 4210) / (4540.0 - 4210.0)
    D2 = (cont + flux_min) / 2.0
    logging.info(f"D2 = {D2:.6f}")

    # Find blue and red edges
    mask_x = wave >= 4210.0
    if np.any(mask_x):
        for i in range(len(wave)):
            if wave[i] >= 4210.0:
                S2 = flux[i] - (D2 + (cont2 - cont1) * (wave_min - 4210) / (4540.0 - 4210.0))
                if S2 <= 0.0:
                    bw = wave[i]
                    break

    mask_x_wavemin = wave >= wave_min
    if np.any(mask_x_wavemin):
        for i in range(len(wave)):
            if wave[i] >= wave_min:
                S2 = flux[i] - (D2 + (cont2 - cont1) * (wave_min - 4210) / (4540.0 - 4210.0))
                if S2 >= 0.0:
                    rw = wave[i]
                    break

    logging.info(f"rw = {rw:.6f}  bw = {bw:.6f}")

    # Check symmetry
    sb = wave_min - bw
    sr = rw - wave_min
    if abs(sb - sr) > 2.0:
        return 0.0
    else:
        D2 = rw - bw
        return D2

def spt_HeII(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Computes the chi-squared value between observed and predicted He II spectral indices.

    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.

    Returns
    -------
    chi2 : float
        The chi-squared value representing the difference between the observed and synthetic He II spectral indices.
        Returns 1e30 if the spectral code is not within the valid range for early B stars or if it falls below the library's `s_hot` threshold.
    """

    if sp_code <= library.s_hot:
        return 1e5

    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    wave_temp, flux_temp = sp2class(library, spectrum_)

    # only for early B stars and earlier #
    if spectrum_.sp_code >= 7.5:
        return 1e30

    # Calculate the sums over the specified ranges
    mask_1 = (wave_temp >= 4511.) & (wave_temp <= 4531.)
    mask_2 = (wave_temp >= 4531.) & (wave_temp <= 4551.)
    mask_3 = (wave_temp >= 4551.) & (wave_temp <= 4571.)
    
    s_sumc1, p_sumc1 = np.sum(flux_temp[mask_1]), np.sum(spectrum_.flux_out[mask_1])
    s_sumI, p_sumI = np.sum(flux_temp[mask_2]), np.sum(spectrum_.flux_out[mask_2])
    s_sumc2, p_sumc2 = np.sum(flux_temp[mask_3]), np.sum(spectrum_.flux_out[mask_3])

    # Calculate the indices
    s_index = s_sumI / 0.5 * (s_sumc1 + s_sumc2)
    p_index = p_sumI / 0.5 * (p_sumc1 + p_sumc2)

    # Calculate chi-squared
    chi2 = (s_index - p_index) ** 2

    return chi2

def hydrogen_index(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Computes the chi-squared (χ²) value representing the difference between the synthetic spectrum and an observed spectrum across specified wavelength ranges. This function is specifically designed to evaluate the spectral type of stars using hydrogen line indices.

    Parameters
    ----------
    sp_code : float
        The spectral type code of the star. Valid values range from 19.0 to 36.0, where values outside this range return 1e30 and 1e10, respectively.
        
    library : Library
        A reference to a `Library` object containing standard stellar spectra data, including wavelength range information.
        
    spectrum : Spectrum
        A `Spectrum` object representing the observed stellar spectrum.

    Returns
    -------
    chi2 : float
        The computed chi-squared (χ²) value, which quantifies the difference between the synthetic and observed spectra. If `sp_code` is outside the valid range, the function returns 1e30 if below and 1e10 if above.
    """
    if sp_code > 36.0: return 1e10
    if sp_code <= 19.0: return 1e30
    
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum_code = spectrum_.lum = spectrum.lum_code

    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    def calculate_chi2(ranges, mults, divs):
        mask = (wave_temp >= ranges[0][0]) & (wave_temp <= ranges[0][1])
        s_sumc1, p_sumc1 = np.sum(flux_temp[mask])/divs[0], np.sum(spectrum.flux_out[mask])/divs[0]
        
        mask = (wave_temp >= ranges[1][0]) & (wave_temp <= ranges[1][1])
        s_sumI, p_sumI = np.sum(flux_temp[mask])/divs[1], np.sum(spectrum.flux_out[mask])/divs[1]
        
        mask = (wave_temp >= ranges[2][0]) & (wave_temp <= ranges[2][1])
        s_sumc2, p_sumc2 = np.sum(flux_temp[mask])/divs[2], np.sum(spectrum.flux_out[mask])/divs[2]
        
        s_indx = s_sumI / (mults[0][0] * s_sumc1 + mults[0][1] * s_sumc2)
        p_indx = p_sumI / (mults[1][0] * p_sumc1 + mults[1][1] * p_sumc2)
        
        chi2 = (s_indx - p_indx) ** 2
        
        return chi2

    chi2 = calculate_chi2([(4052.0, 4072.0), (4082.0, 4122.0), (4132.0, 4152.0)], [(0.5, 0.5), (0.5, 0.5)], [20, 40, 20])

    chi2 += calculate_chi2([(4233.0, 4248.0), (4320.0, 4360.0), (4355.0, 4378.0)], [(0.206, 0.794), (0.206, 0.794)], [15, 40, 23])

    if library.w_high > 4920:
        chi2 += calculate_chi2([(4805.0, 4825.0), (4841.0, 4881.0), (4897.0, 4917.0)], [(0.5, 0.5), (0.206, 0.794)], [20, 40, 20])
    
    return chi2

def hydrogen_profile(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Compares hydrogen-line profiles with a rough rectification routine to accommodate flux-calibrated spectra.

    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.

    Returns
    -------
    chi2 : float
        The chi-squared value comparing the star's hydrogen-line profiles.
    """
    # Only consider A-stars and later
    if sp_code <= 19.0: return 1e30

    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    def calculate_chi2(x_ranges):
        mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
        s_sumc1, p_sumc1 = np.mean(flux_temp[mask]), np.mean(spectrum_.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])
        s_sumc2, p_sumc2 = np.mean(flux_temp[mask]), np.mean(spectrum_.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1])
        s_cont = s_sumc1 + (s_sumc2 - s_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        p_cont = p_sumc1 + (p_sumc2 - p_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        
        chi2 = np.sum(((flux_temp[mask] / s_cont) - (spectrum_.flux_out[mask] / p_cont)) ** 2)
        
        return chi2
        
    # H-delta line
    chi2 = calculate_chi2([(4012.0, 4052.0), (4152.0, 4192.0), (4062.0, 4142.0), (4032.0, 4172.0)])

    # H-gamma line
    chi2 += calculate_chi2([(4250.0, 4290.0), (4390.0, 4430.0), (4300.0, 4380.0), (4270.0, 4410.0)])

    # H-beta line, only if w_high > 4960 Å
    if library.w_high > 4960:
        chi2 += calculate_chi2([(4791.0, 4831.0), (4911.0, 4951.0), (4841.0, 4901.0), (4811.0, 4931.0)])

    return chi2

def hydrogen_profile_hot(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Compares hydrogen-line profiles.
    
    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.

    Returns
    -------
    chi2 : float
        The chi-squared value representing the difference between the observed and synthetic spectral indices.
    """
    if sp_code >= 17.0:
        return 1e30  # Only for B-stars and earlier

    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum_code = spectrum_.lum = spectrum.lum_code

    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    def calculate_chi2(x_ranges):
        mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
        s_sumc1, p_sumc1 = np.mean(flux_temp[mask]), np.mean(spectrum_.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])
        s_sumc2, p_sumc2 = np.mean(flux_temp[mask]), np.mean(spectrum_.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1])
        s_cont = s_sumc1 + (s_sumc2 - s_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        p_cont = p_sumc1 + (p_sumc2 - p_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        
        chi2 = np.sum(((flux_temp[mask] / s_cont) - (spectrum_.flux_out[mask] / p_cont)) ** 2)
        
        return chi2

    # H-delta line
    chi2 = calculate_chi2([(4012.0, 4052.0), (4152.0, 4192.0), (4062.0, 4142.0), (4032.0, 4172.0)])

    # H-gamma line
    chi2 += calculate_chi2([(4250.0, 4290.0), (4390.0, 4430.0), (4300.0, 4380.0), (4270.0, 4410.0)])

    if library.w_high > 4960:
        # H-beta line
        chi2 += calculate_chi2([(4791.0, 4831.0), (4911.0, 4951.0), (4841.0, 4901.0), (4811.0, 4931.0)])

    return chi2

def spt_HeImet(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Calculates the spectral type based on He I strengths and other temperature-sensitive criteria.
    
    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.
    
    Returns
    -------
    chi2 : float
        The chi-squared value indicating the match of the spectral type to standards.
    """
    # Only valid for B stars and earlier
    if sp_code >= 20.0:
        return 1e10
    
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum_code = spectrum_.lum = spectrum.lum_code
    
    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    # Placeholder for the chi-squared value
    chi2 = hydrogen_profile_hot(sp_code, library, spectrum_)
    
    ###################################################################
    ### Calculate chi2 contributions from various spectral features ###
    ###################################################################
    
    # He I 4026 line
    mask = (wave_temp >= 4014) & (wave_temp <= 4019)
    ssumc1, psumc1 = np.sum(flux_temp[mask])/5, np.sum(spectrum.flux_out[mask])/5
    
    mask = (wave_temp >= 4019) & (wave_temp <= 4033)
    ssumI, psumI = np.sum(flux_temp[mask])/14, np.sum(spectrum.flux_out[mask])/14
    
    mask = (wave_temp >= 4033) & (wave_temp <= 4038)
    ssumc2, psumc2 = np.sum(flux_temp[mask])/5, np.sum(spectrum.flux_out[mask])/5
    
    sindx = ssumI / (0.5*ssumc1 + 0.5*ssumc2)
    pindx = psumI / (0.5*psumc1 + 0.5*psumc2)
    chi2 += (sindx - pindx)**2
    
    # He I 4387
    mask = (wave_temp >= 4365) & (wave_temp <= 4377)
    ssumc1, psumc1 = np.sum(flux_temp[mask])/12, np.sum(spectrum.flux_out[mask])/12
    
    mask = (wave_temp >= 4377) & (wave_temp <= 4397)
    ssumI, psumI = np.sum(flux_temp[mask])/20, np.sum(spectrum.flux_out[mask])/20
    
    mask = (wave_temp >= 4397) & (wave_temp <= 4409)
    ssumc2, psumc2 = np.sum(flux_temp[mask])/12, np.sum(spectrum.flux_out[mask])/12
    
    sindx = ssumI / (0.5*ssumc1 + 0.5*ssumc2)
    pindx = psumI / (0.5*psumc1 + 0.5*psumc2)
    chi2 += (sindx - pindx)**2
    
    # He I 4471
    mask = (wave_temp >= 4439) & (wave_temp <= 4454)
    ssumc1, psumc1 = np.sum(flux_temp[mask])/15, np.sum(spectrum.flux_out[mask])/15
    
    mask = (wave_temp >= 4465) & (wave_temp <= 4476)
    ssumI, psumI = np.sum(flux_temp[mask])/11, np.sum(spectrum.flux_out[mask])/11
    
    mask = (wave_temp >= 4486) & (wave_temp <= 4501)
    ssumc2, psumc2 = np.sum(flux_temp[mask])/15, np.sum(spectrum.flux_out[mask])/15
    
    sindx = ssumI / (0.5*ssumc1 + 0.5*ssumc2)
    pindx = psumI / (0.5*psumc1 + 0.5*psumc2)
    chi2 += (sindx - pindx)**2
    
    # Si II 4128-30 ratioed with He I 4144
    mask = (wave_temp >= 4124) & (wave_temp <= 4135)
    ssumc1, psumc1 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= 4136) & (wave_temp <= 4150)
    ssumc2, psumc2 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    ratio1, ratio2 = ssumc1/ssumc2, psumc1/psumc2
    chi2 += 2 * (ratio1 - ratio2)**2
    
    # The feature at 4071 in ratio with nearby continuum
    mask = (wave_temp >= 4053) & (wave_temp <= 4089)
    ssumc1, psumc1 = np.sum(flux_temp[mask])/36, np.sum(spectrum.flux_out[mask])/36
    
    mask = (wave_temp >= 4065) & (wave_temp <= 4080)
    ssumc2, psumc2 = np.sum(flux_temp[mask])/15, np.sum(spectrum.flux_out[mask])/15
    
    ratio1, ratio2 = ssumc1/ssumc2, psumc1/psumc2
    chi2 += 2 * (ratio1 - ratio2)**2
    
    # The C II 4267 line in ratio with nearby continua
    mask = (wave_temp >= 4248) & (wave_temp <= 4260)
    ssumc1, psumc1 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= 4260) & (wave_temp <= 4272)
    ssumI, psumI = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= 4272) & (wave_temp <= 4284)
    ssumc2, psumc2 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    sindx = ssumI / (0.5*ssumc1 + 0.5*ssumc2)
    pindx = psumI / (0.5*psumc1 + 0.5*psumc2)
    chi2 += 2 * (sindx - pindx)**2
    
    # The ratio of He I 4471 to Mg II 4481
    # The feature at 4071 in ratio with nearby continuum
    mask = (wave_temp >= 4463) & (wave_temp <= 4475)
    ssumc1, psumc1 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= 4475) & (wave_temp <= 4486)
    ssumc2, psumc2 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    ratio1, ratio2 = ssumc1/ssumc2, psumc1/psumc2
    chi2 += 5 * (ratio1 - ratio2)**2
    
    return chi2

def spt_CaK(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Calculates the chi-squared value based on the Ca II K line region.

    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.

    Returns
    -------
    chi2 : float
        The calculated chi-squared value.
    """
    if sp_code < 13.0:
        return 10.0

    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    # Compute s_sum and p_sum for the range [3918.0, 3925.0]
    mask = (wave_temp >= 3918.0) & (wave_temp <= 3925.0)
    s_sum = np.sum(flux_temp[mask]) / 8.0
    p_sum = np.sum(spectrum_.flux_out[mask]) / 8.0
    
    # Calculate chi-squared in the range [3927.0, 3937.0]
    mask = (wave_temp >= 3927.0) & (wave_temp <= 3937.0)
    chi2 = np.sum(((flux_temp[mask] / s_sum) - (spectrum_.flux_out[mask] / p_sum)) ** 2)
    
    return chi2

def spt_KM(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Computes the chi-squared value comparing the observed spectrum with a template spectrum.

    Parameters:
    -----------
    sp_code : float
        The spectral classification code, where values <= 39 correspond to early K-type stars.
        
    library : Library
        The library containing template spectra for different star types.
        
    spectrum : Spectrum
        The observed spectrum to be classified.

    Returns:
    --------
    chi2 : float
        The chi-squared value representing the goodness-of-fit between the observed and template spectra.
    """
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum_code = spectrum_.lum = spectrum.lum_code
    
    wave_temp, flux_temp = sp2class(library, spectrum_)
    
    mask_ch2 = (wave_temp >= 3900) & (wave_temp <= 4250)
    ch2 = np.sum((spectrum.flux_out[mask_ch2] - flux_temp[mask_ch2]) ** 2) / 350
    
    chi2 = 0.
    # Early K-type star temperature classification #
    if sp_code <= 39.0:
        # Ca I, G-band indices
        mask_stdc1 = (wave_temp >= 4202.5) & (wave_temp <= 4220.5)
        mask_stdCaI = (wave_temp >= 4225.5) & (wave_temp <= 4227.5)
        mask_stdc2 = (wave_temp >= 4233.0) & (wave_temp <= 4248.0)
        mask_stdG = (wave_temp >= 4297.0) & (wave_temp <= 4314.0)
        mask_stdc3 = (wave_temp >= 4355.0) & (wave_temp <= 4378.0)

        stdc1 = np.sum(flux_temp[mask_stdc1]) / 18.0
        proc1 = np.sum(spectrum.flux_out[mask_stdc1]) / 18.0
        stdCaI = np.sum(flux_temp[mask_stdCaI]) / 2.0
        proCaI = np.sum(spectrum.flux_out[mask_stdCaI]) / 2.0
        stdc2 = np.sum(flux_temp[mask_stdc2]) / 15.0
        proc2 = np.sum(spectrum.flux_out[mask_stdc2]) / 15.0
        stdG = np.sum(flux_temp[mask_stdG]) / 17.0
        proG = np.sum(spectrum.flux_out[mask_stdG]) / 17.0
        stdc3 = np.sum(flux_temp[mask_stdc3]) / 15.0
        proc3 = np.sum(spectrum.flux_out[mask_stdc3]) / 15.0

        istdG = stdG / (0.484 * stdc2 + 0.516 * stdc3)
        iproG = proG / (0.484 * proc2 + 0.516 * proc3)
        istdCaI = stdCaI / (stdc1 + stdc2)
        iproCaI = proCaI / (proc1 + proc2)

        chi2 += ((1.0 - istdG / iproG) ** 2 + (1.0 - istdCaI / iproCaI) ** 2)
        
        # Mg I index
        mask_stdc1 = (wave_temp >= 5004.0) & (wave_temp <= 5058.0)
        mask_stdMgI = (wave_temp >= 5158.0) & (wave_temp <= 5212.0)
        mask_stdc2 = (wave_temp >= 5312.0) & (wave_temp <= 5366.0)
        
        stdc1 = np.sum(flux_temp[mask_stdc1])
        proc1 = np.sum(spectrum.flux_out[mask_stdc1])
        stdMgI = np.sum(flux_temp[mask_stdMgI])
        proMgI = np.sum(spectrum.flux_out[mask_stdMgI])
        stdc2 = np.sum(flux_temp[mask_stdc2])
        proc2 = np.sum(spectrum.flux_out[mask_stdc2])

        istdMgI = stdMgI / (stdc1 + stdc2)
        iproMgI = proMgI / (proc1 + proc2)
        
        if iproMgI != 0.0:
            chi2 += 2.0 * (1.0 - istdMgI / iproMgI) ** 2

        # MgH band
        mask_stdc1 = (wave_temp >= 4722.0) & (wave_temp <= 4750.0)
        mask_stdMgH = (wave_temp >= 4750.0) & (wave_temp <= 4791.0)
        mask_stdc2 = (wave_temp >= 4791.0) & (wave_temp <= 4816.0)

        stdc1 = np.sum(flux_temp[mask_stdc1])
        proc1 = np.sum(spectrum.flux_out[mask_stdc1])
        stdMgH = np.sum(flux_temp[mask_stdMgH])
        proMgH = np.sum(spectrum.flux_out[mask_stdMgH])
        stdc2 = np.sum(flux_temp[mask_stdc2])
        proc2 = np.sum(spectrum.flux_out[mask_stdc2])

        istdMgH = stdMgH / (stdc1 + stdc2)
        iproMgH = proMgH / (proc1 + proc2)

        if iproMgH != 0.0:
            chi2 += 2.0 * (1.0 - istdMgH / iproMgH) ** 2

        chi2 /= 6.0
    
    else:
        # Mg I index
        mask_stdc1 = (wave_temp >= 5004.0) & (wave_temp <= 5058.0)
        mask_stdMgI = (wave_temp >= 5158.0) & (wave_temp <= 5212.0)
        mask_stdc2 = (wave_temp >= 5312.0) & (wave_temp <= 5366.0)
        
        stdc1 = np.sum(flux_temp[mask_stdc1])
        proc1 = np.sum(spectrum.flux_out[mask_stdc1])
        stdMgI = np.sum(flux_temp[mask_stdMgI])
        proMgI = np.sum(spectrum.flux_out[mask_stdMgI])
        stdc2 = np.sum(flux_temp[mask_stdc2])
        proc2 = np.sum(spectrum.flux_out[mask_stdc2])

        istdMgI = stdMgI / (stdc1 + stdc2)
        iproMgI = proMgI / (proc1 + proc2)
        
        if iproMgI != 0.0:
            chi2 += 2.0 * (1.0 - istdMgI / iproMgI) ** 2

        # MgH band
        mask_stdc1 = (wave_temp >= 4722.0) & (wave_temp <= 4750.0)
        mask_stdMgH = (wave_temp >= 4750.0) & (wave_temp <= 4791.0)
        mask_stdc2 = (wave_temp >= 4791.0) & (wave_temp <= 4816.0)

        stdc1 = np.sum(flux_temp[mask_stdc1])
        proc1 = np.sum(spectrum.flux_out[mask_stdc1])
        stdMgH = np.sum(flux_temp[mask_stdMgH])
        proMgH = np.sum(spectrum.flux_out[mask_stdMgH])
        stdc2 = np.sum(flux_temp[mask_stdc2])
        proc2 = np.sum(spectrum.flux_out[mask_stdc2])

        istdMgH = stdMgH / (stdc1 + stdc2)
        iproMgH = proMgH / (proc1 + proc2)

        if iproMgH != 0.0:
            chi2 += 2.0 * (1.0 - istdMgH / iproMgH) ** 2
        
        # CaOH band
        mask_stdc1 = (wave_temp >= 5499.0) & (wave_temp <= 5524.0)
        mask_stdCaOH = (wave_temp >= 5532.5) & (wave_temp <= 5548.0)
        
        stdc1 = np.sum(flux_temp[mask_stdc1])
        proc1 = np.sum(spectrum.flux_out[mask_stdc1])
        stdCaOH = np.sum(flux_temp[mask_stdCaOH])
        proCaOH = np.sum(spectrum.flux_out[mask_stdCaOH])

        istdCaOH = stdCaOH / stdc1
        iproCaOH = proCaOH / proc1

        if iproCaOH != 0.0:
            chi2 += 2.0 * (1.0 - istdCaOH / iproCaOH) ** 2
        
        # TiO band head measures
        masks = [
            ((wave_temp >= 4731.0) & (wave_temp <= 4751.0), (wave_temp >= 4766.0) & (wave_temp <= 4786.0)),
            ((wave_temp >= 4929.0) & (wave_temp <= 4949.0), (wave_temp >= 4959.0) & (wave_temp <= 4979.0)),
            ((wave_temp >= 5140.0) & (wave_temp <= 5160.0), (wave_temp >= 5172.0) & (wave_temp <= 5192.0)),
            ((wave_temp >= 5424.0) & (wave_temp <= 5444.0), (wave_temp >= 5456.0) & (wave_temp <= 5476.0)),
        ]

        for mask1, mask2 in masks:
            std1 = np.sum(flux_temp[mask1])
            pro1 = np.sum(spectrum.flux_out[mask1])
            std2 = np.sum(flux_temp[mask2])
            pro2 = np.sum(spectrum.flux_out[mask2])

            if std2 != 0.0 and pro2 != 0.0:
                chi2 += (1.0 - (std1 / std2) / (pro1 / pro2)) ** 2

        chi2 /= 8.0
    
    return chi2 + ch2

def hydrogen_luminosity(spectrum, wave_temp, flux_temp) -> float:
    """
    Calculates the chi-squared statistic for hydrogen lines (H-delta, H-gamma, H-beta) by comparing a target spectrum with a template spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        An object representing the observed spectrum.
            
    wave_temp : numpy.ndarray
        A 1D numpy array of wavelength values (in angstroms) corresponding to the observed and template spectra.
        
    flux_temp : numpy.ndarray
        A 1D numpy array of flux values representing the template spectrum corresponding to the wavelength values in `wave_temp`.

    Returns
    -------
    chi2 : float
        The chi-squared statistic representing the cumulative difference between the observed 
        spectrum and the template spectrum across the H-delta, H-gamma, and H-beta lines.
    """
    def calculate_chi2(x_ranges):
        mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
        s_sumc1, p_sumc1 = np.mean(flux_temp[mask]), np.mean(spectrum.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])
        s_sumc2, p_sumc2 = np.mean(flux_temp[mask]), np.mean(spectrum.flux_out[mask])
        
        mask = (wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1])
        s_cont = s_sumc1 + (s_sumc2 - s_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        p_cont = p_sumc1 + (p_sumc2 - p_sumc1) * (wave_temp[mask] - x_ranges[3][0]) / (x_ranges[3][1] - x_ranges[3][0])
        
        chi2 = np.sum(((flux_temp[mask] / s_cont) - (spectrum.flux_out[mask] / p_cont)) ** 2)
        
        return chi2

    # H-delta line
    chi2 = calculate_chi2([(4012.0, 4052.0), (4152.0, 4192.0), (4062.0, 4142.0), (4032.0, 4172.0)])

    # H-gamma line
    chi2 += calculate_chi2([(4250.0, 4290.0), (4390.0, 4430.0), (4300.0, 4380.0), (4270.0, 4410.0)])

    # H-beta line
    chi2 += calculate_chi2([(4791.0, 4831.0), (4911.0, 4951.0), (4841.0, 4901.0), (4811.0, 4931.0)])

    return chi2

def lum_ratio_min(lum_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Computes the chi-squared statistic to determine the best-fit luminosity class for a given spectrum by comparing it to a template library spectrum across various spectral lines.

    Parameters
    ----------
    lum_code : float
        The luminosity code to be set for the input `spectrum` object before performing the chi-squared comparison.
        
    library : Library
        A reference object containing the template spectra against which the input `spectrum` will be compared.
        
    spectrum : Spectrum
        The observed spectrum to be compared with the template spectra.

    Returns
    -------
    chi2 : float
        The chi-squared statistic representing the cumulative difference between the observed spectrum and the template spectra, adjusted for various spectral line ratios.

    Notes
    -----
    - The spectral lines analyzed include:
        - Hydrogen lines (H-delta, H-gamma, H-beta) for A-type stars
        - O II, Si III, and He I lines for hotter stars (B-type)
        - Fe II, Ti II, and Sr II lines for F and G-type stars
        - Sr II and CN band for G and K-type stars
        - MgH and Mg I bands for cooler stars (late K-type and M-type)

    - The function considers special cases, such as Barium stars (`Ba` attribute of the spectrum), where certain spectral lines are excluded from the luminosity determination.
    """
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.lum_code = spectrum_.lum = lum_code
    spectrum_.spt = spectrum.sp_code
    
    wave_temp, flux_temp = sp2class(library, spectrum_)

    def calculate_chi2(x_ranges, mode):
        if mode == 1:
            mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
            std1, pro1 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
            
            mask = (wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])
            std2, pro2 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
        elif mode == 2:
            mask = ((wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])) | ((wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1]))
            std1, pro1 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
            
            mask = (wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1])
            std2, pro2 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
        elif mode == 3:
            mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
            std1, pro1 = 9.0 * np.sum(flux_temp[mask]), 9.0 * np.sum(spectrum_.flux_out[mask])
            
            mask = ((wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])) | ((wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1]))
            std2, pro2 = 7.0 * np.sum(flux_temp[mask]), 7.0 * np.sum(spectrum_.flux_out[mask])
        elif mode == 4:
            mask = (wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])
            std1, pro1 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
            
            mask = ((wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])) | ((wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1]))
            std2, pro2 = np.sum(flux_temp[mask]), np.sum(spectrum_.flux_out[mask])
        else:
            return 0.0
        
        rat_std, rat_pro = std1 / std2, pro1 / pro2
        if rat_pro == 0.0: return 0.0
        chi2 = (1.0 - rat_std / rat_pro) ** 2
        
        return chi2
    
    chi2 = 0.0
    if 7.0 <= spectrum_.sp_code < 10.0:
        chi2 = hydrogen_luminosity(spectrum_, wave_temp, flux_temp)
        """
        We also use with heavier weight luminosity criteria based on O II lines and Si III lines ratioed to He I lines
        """
        chi2 += 4.0 * (calculate_chi2([(4379.0, 4395.0), (4407.0, 4422.0)], 1) + calculate_chi2([(4463.0, 4476.0), (4545.0, 4581.0)], 1))

    if 10.0 <= spectrum_.sp_code < 20.5:
        # A-type star luminosity criteria -- Hydrogen lines
        chi2 = hydrogen_luminosity(spectrum_, wave_temp, flux_temp)
    
    elif spectrum_.sp_code < 27.5:
        # Early F-star luminosity criteria -- FeII TiII lines
        chi2 = calculate_chi2([(4166.0, 4180.0), (4146.0, 4162.0)], 1) + calculate_chi2([(4410.0, 4419.0), (4438.0, 4448.0), (4419.0, 4438.0)], 2)
    
    elif spectrum_.sp_code < 31.5:
        # Late F and early G luminosity criteria -- FeII, TiII, SrII lines
        chi2 = calculate_chi2([(4166.0, 4180.0), (4146.0, 4162.0)], 1) + calculate_chi2([(4410.0, 4419.0), (4438.0, 4448.0), (4419.0, 4438.0)], 2)
        
        # Don't include the Sr II lines in the luminosity determination if the star is a Barium star
        if not spectrum_.Ba:
            chi2 += calculate_chi2([(4073.0, 4082.0), (4041.0, 4058.0)], 1) + calculate_chi2([(4212.0, 4218.0), (4220.0, 4230.0)], 1)

    elif spectrum_.sp_code < 39.0:
        chi2 = 0.0
        # Late G and early K-type luminosity criteria -- Sr II
        if not spectrum_.Ba:
            chi2 += calculate_chi2([(4073.0, 4082.0), (4041.0, 4058.0)], 1) + calculate_chi2([(4212.0, 4218.0), (4220.0, 4230.0)], 1)
        
        # 4215 CN band
        chi2 += 3.0 * calculate_chi2([(4140.0, 4210.0), (4219.0, 4264.0), (4043.0, 4088.0)], 3)
        
        if spectrum_.sp_code >= 32.0 and library.w_high > 5278.0:
            # 5250/5269
            chi2 += 5.0 * calculate_chi2([(5239.0, 5256.0), (5256.0, 5278.0)], 1)

    elif spectrum_.sp_code >= 39.:
        chi2 = 0.0
        # The 5050 region
        # chi2 += 0.0 * calculate_chi2([(4980.0, 5030.0), (5100.0, 5150.0)], 1)
        
        # The MgH band
        chi2 += calculate_chi2([(4760.0, 4785.0), (4720.0, 4745.0)], 1)
        
        # 5250/5269
        chi2 += 5.0 * calculate_chi2([(5239.0, 5256.0), (5256.0, 5278.0)], 1)
        
        # MgI band
        chi2 += 2.0 * calculate_chi2([(5158.0, 5212.0), (5004.0, 5058.0), (5312.0, 5366.0)], 4)
    
    return chi2

def HeI_pec(library: Library, spectrum: Spectrum) -> float:
    """
    Evaluates the helium peculiarity in B-type stars based on the provided spectrum.

    Parameters
    ----------
    library : Library
        A reference object containing the template spectra against which the input `spectrum` will be compared.
        
    spectrum : Spectrum
        The observed spectrum to be compared with the template spectra.
        
    Returns
    -------
    dif : float
        The helium peculiarity index. If `dif` < 0, helium is weak; if `dif` ~ 0, helium is normal; if `dif` > 0, helium is strong.
        If the spectral type is not B-type (`sp_code` >= 19.0), returns 1e30.
    """

    if spectrum.sp_code >= 19.0:
        return 1e30  # Only for B-type stars and earlier

    wave_temp, flux_temp = sp2class(library, spectrum)
    mask = (wave_temp >= 4019.0) & (wave_temp <= 4033.0)
    s_sumI = np.mean(flux_temp[mask])
    p_sumI = np.mean(spectrum.flux_out[mask])

    # Calculate the helium peculiarity index
    dif = (s_sumI - p_sumI) / s_sumI

    return dif

def spt_metal(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Calculates chi-squared for metallic-line type. Mostly useful for F and G-type stars.

    Parameters
    ----------
    sp_code : float
        The spectral code representing the spectral type of the star being analyzed.
        
    library : Library
        An instance of the `Library` class that contains various spectral properties and thresholds.
        
    spectrum : Spectrum
        An instance of the `Spectrum` class that contains the observed spectrum data including wavelength and flux arrays.

    Returns
    -------
    chi2 : float
        Chi-squared value for metallic-line type.
    """
    from .peculiarities import CH_band2
    
    if sp_code > 36.0: return 1e10

    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum = spectrum.lum_code
    wave_temp, flux_temp = sp2class(library, spectrum_)

    # Define spectral regions for metallic-line comparisons
    regions = [(4135.0, 4315.0), (4410.0, 4600.0)]

    chi2a = 0.0
    n = 0
    
    # Loop through spectral regions
    for start, end in regions:
        mask = (wave_temp >= start) & (wave_temp <= end)
        if not any(mask):
            continue
        
        pa, pb = lst_sqr(spectrum_.wave_out, spectrum_.flux_out, start, end)
        sa, sb = lst_sqr(wave_temp, flux_temp, start, end)

        chi2a += np.sum((flux_temp[mask] / (sa + wave_temp[mask] * sb) - spectrum_.flux_out[mask] / (pa + wave_temp[mask] * pb)) ** 2)
        n += np.sum(mask)
    
    mask = (wave_temp >= 4900.0) & (wave_temp <= 5400.0)
    if any(mask):
        pa, pb = lst_sqr(spectrum_.wave_out, spectrum_.flux_out, 4900.0, 5400.0)
        sa, sb = lst_sqr(wave_temp, flux_temp, 4900.0, 5400.0)

        chi2a += np.sum((flux_temp[mask] / (sa + wave_temp[mask] * sb) - spectrum_.flux_out[mask] / (pa + wave_temp[mask] * pb)))

    chi2a /= max(n, 1)

    # Add CH band difference
    chi = CH_band2(sp_code, spectrum_.lum_code, library, spectrum_)
    # logging.info(f"CH band dif = {chi:.6f}")
    chi2 = chi2a + 3.0 * chi ** 2

    return chi2

def ratio_CaI_FeII(wave: np.ndarray, flux: np.ndarray) -> float:
    """
    Calculates the ratio of the Ca I 4226 line to the Fe II 4233 line.

    Parameters
    ----------
    wave : np.ndarray
        The wavelength array.
        
    flux : np.ndarray
        The flux array corresponding to the wavelengths in X.

    Returns
    -------
    ratio : float
        The ratio of the Ca I 4226 line to the Fe II 4233 line.
    """
    # Define wavelength ranges
    cont_mask = ((wave >= 4219.0) & (wave <= 4222.0)) | ((wave > 4242.0) & (wave <= 4245.0))
    p_sumc1_mask = (wave >= 4223.0) & (wave <= 4230.0)
    p_sumc2_mask = (wave >= 4231.0) & (wave <= 4235.0)

    # Calculate the continuum level
    cont = np.mean(flux[cont_mask])

    # Calculate the p_sumc1 and p_sumc2 values
    p_sumc1 = np.mean(cont - flux[p_sumc1_mask])
    p_sumc2 = np.mean(cont - flux[p_sumc2_mask])

    # Calculate and return the ratio
    ratio = p_sumc1 / p_sumc2
    return ratio

def spt_G_lines(sp_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Calculates a chi-squared (χ²) value based on the ratio of hydrogen to iron lines in late F and G-type stars to determine the temperature type.

    Parameters
    ----------
    sp_code : float
        The spectral type code of the star. Should be >= 28.0 for meaningful results.
        
    library : Library
        A reference to a `Library` object containing stellar spectra data, including wavelength range information.
        
    spectrum : Spectrum
        A `Spectrum` object representing the observed stellar spectrum.

    Returns
    -------
    chi2 : float
        The computed chi-squared (χ²) value based on the hydrogen and iron line ratios. If `sp_code` is less than 28.0, the function returns 0.0.
    """

    if sp_code < 28.0: return 0.0
    
    # Deep copy the spectrum and set the spectral type code
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code

    # Generate the synthetic spectrum for the given spectral type code
    wave_temp, flux_temp = sp2class(library, spectrum_)

    def calculate_chi2(wave_ranges):
        """Helper function to calculate chi-squared contribution from a set of wavelength ranges."""
        mask_1 = (wave_temp > wave_ranges[0][0]) & (wave_temp < wave_ranges[0][1])
        mask_2 = (wave_temp > wave_ranges[1][0]) & (wave_temp < wave_ranges[1][1])
        std_1 = np.sum(flux_temp[mask_1])
        pro_1 = np.sum(spectrum.flux_out[mask_1])
        std_2 = np.sum(flux_temp[mask_2])
        pro_2 = np.sum(spectrum.flux_out[mask_2])
        
        if std_2 != 0.0 and pro_2 != 0.0:
            rat_std = std_1 / std_2
            rat_pro = pro_1 / pro_2
            return (1.0 - rat_std / rat_pro) ** 2
        return 0.0

    # H delta and Fe I 4046
    chi2 = calculate_chi2([(4038.0, 4050.0), (4096.0, 4106.0)])

    # H gamma and Fe I 4383 (weighted contribution)
    chi2 += 3.0 * calculate_chi2([(4378.0, 4388.0), (4332.0, 4347.0)])

    # H beta and metal 4888 (only if `w_high` is sufficiently high)
    if library.w_high > 4900.0:
        chi2 += calculate_chi2([(4880.0, 4894.0), (4851.0, 4866.0)])

    return chi2

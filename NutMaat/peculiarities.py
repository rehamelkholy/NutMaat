import logging
import numpy as np

from .library import *
from .spectrum import *
from .evaluate import *
from .utils import *

def peculiarity(library: Library, spectrum: Spectrum) -> bool:
    """
    Determines the peculiarity of various elements (Sr, Eu, Si) in a star's spectrum based on its spectral and luminosity codes.

    Parameters
    ----------
    library : Library
        An object containing the library of standard spectra and relevant parameters.

    spectrum : Spectrum
        An object representing the input spectrum.

    Returns
    -------
    flag : bool
        The peculiarity flag indicating whether any was found.

    Notes
    -----
    - Strontium peculiarity is checked for stars with spectral types from A0 to K0 (`sp_code` between 16.0 and 34.0).
    - Europium peculiarity is checked for stars with spectral types from A0 to F5 (`sp_code` between 16.0 and 26.0).
    - Silicon peculiarity is checked for stars with spectral types from B5 to F0 (`sp_code` between 12.0 and 23.0).
    - If a peculiarity is detected, the corresponding `spectrum` attribute key will be set to True.
    """

    wave_temp, flux_temp = sp2class(library, spectrum)

    def calculate_diff(x_ranges):
        """Helper function to calculate the ratio of flux in two wavelength ranges."""
        mask_1 = (wave_temp >= x_ranges[1][0]) & (wave_temp <= x_ranges[1][1])
        mask_2 = ((wave_temp >= x_ranges[0][0]) & (wave_temp <= x_ranges[0][1])) | ((wave_temp >= x_ranges[2][0]) & (wave_temp <= x_ranges[2][1]))
        
        s_sum1, p_sum1 = np.sum(flux_temp[mask_1]), np.sum(spectrum.flux_out[mask_1])
        s_sum2, p_sum2 = np.sum(flux_temp[mask_2]), np.sum(spectrum.flux_out[mask_2])
        
        s_ratio, p_ratio = s_sum1 / (0.5 * s_sum2), p_sum1 / (0.5 * p_sum2)
        
        dif = (s_ratio - p_ratio) / s_ratio
        return dif

    # Strontium Peculiarity: from A0 to K0
    if 16.0 <= spectrum.spt <= 34.0:
        dif = calculate_diff([(4068.0, 4074.5), (4074.5, 4081.0), (4081.0, 4087.5)])
        logging.info(f"Sr II: dif = {dif:.6f}")
        if 0.015 <= dif <= 0.024:
            spectrum.Sr = 1
            spectrum.pec = True
        elif dif > 0.024:
            spectrum.Sr = 2
            spectrum.pec = True

    # Europium Peculiarity: from A0 to F5
    if 16.0 <= spectrum.spt <= 26.0:
        dif = calculate_diff([(4199.0, 4203.0), (4203.0, 4207.0), (4207.0, 4211.0)])
        if dif >= 0.015:
            spectrum.Eu = 1
            spectrum.pec = True

    # Silicon Peculiarity: from B5 to F0
    if 12.0 <= spectrum.spt <= 23.0:
        dif = calculate_diff([(4118.0, 4126.0), (4126.0, 4134.0), (4134.0, 4142.0)])
        logging.info(f"Si II: dif = {dif:.6f}")
        if dif >= 0.015:
            spectrum.Si = 1
            spectrum.pec = True

    return spectrum.pec

def MgII(library: Library, spectrum: Spectrum) -> float:
    """
    Measures the Mg II index based on the Mg II 4481 line to indicate a possible Lambda Boo star.

    Parameters
    ----------
    library : Library
        An object containing the library of standard spectra and relevant parameters.

    spectrum : Spectrum
        An object representing the input spectrum.

    Returns
    -------
    mg : float
        The Mg II index, where a positive value suggests a possible Lambda Boo star.
    """
    # Only consider stars with spcode <= 34.0
    if spectrum.sp_code > 34.0: return 0.0
    
    wave_temp, flux_temp = sp2class(library, spectrum)
    
    mask = (wave_temp >= 4478.0) & (wave_temp <= 4484.0)
    s_sum1, p_sum1 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= 4484.0) & (wave_temp <= 4490.0)
    s_sum2, p_sum2 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
    
    s_ratio, p_ratio = s_sum1 / s_sum2, p_sum1 / p_sum2

    # Compute the Mg II index
    mg = (s_ratio - p_ratio) / s_ratio

    return mg

def carbon_4737(library: Library, spectrum: Spectrum) -> float:
    """
    Determines whether the C2 4737 band is enhanced in a given spectrum.

    Parameters
    ----------
    library : Library
        An object representing the spectral library that provides reference spectra for comparison.
        
    spectrum : Spectrum
        An object representing the observed spectrum of the star.

    Returns
    -------
    dif : float
        The difference between the ratios of the observed and reference spectra. A higher difference suggests potential enhancement of the C2 4737 band.
    """
    wave_temp, flux_temp = sp2class(library, spectrum)

    ranges = [
        (4585.0, 4630.0),
        (4630.0, 4720.0),
        (4720.0, 4765.0)
    ]

    mask = ((wave_temp >= ranges[0][0]) & (wave_temp <= ranges[0][1])) | ((wave_temp >= ranges[2][0]) & (wave_temp <= ranges[2][1]))
    s_sum1 = np.sum(flux_temp[mask])
    p_sum1 = np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= ranges[1][0]) & (wave_temp <= ranges[1][1])
    s_sum2 = np.sum(flux_temp[mask])
    p_sum2 = np.sum(spectrum.flux_out[mask])
    
    if s_sum2 == 0 or p_sum2 == 0: return 0.
    
    ratio_1 = s_sum1 / s_sum2
    ratio_2 = p_sum1 / p_sum2
    
    dif = abs((ratio_1 - ratio_2)) / ratio_2

    return dif

def CN_4215(library: Library, spectrum: Spectrum) -> float:
    """
    Checks if the CN 4215 band is enhanced by comparing the spectrum to a reference library of spectra.

    Parameters
    ----------
    library : Library
        The reference library of spectra.
        
    spectrum : Spectrum
        The spectrum to be analyzed.

    Returns
    -------
    dif : float
        The calculated difference indicating the enhancement of the CN 4215 band.
        Returns 0.0 if the calculation could not be performed.
    """
    
    wave_temp, flux_temp = sp2class(library, spectrum)
    
    ranges = [
        (4043.0, 4088.0),
        (4140.0, 4210.0),
        (4219.0, 4264.0)
    ]

    mask = ((wave_temp >= ranges[0][0]) & (wave_temp <= ranges[0][1])) | ((wave_temp >= ranges[2][0]) & (wave_temp <= ranges[2][1]))
    s_sum1 = np.sum(flux_temp[mask])
    p_sum1 = np.sum(spectrum.flux_out[mask])
    
    mask = (wave_temp >= ranges[1][0]) & (wave_temp <= ranges[1][1])
    s_sum2 = np.sum(flux_temp[mask])
    p_sum2 = np.sum(spectrum.flux_out[mask])
    
    if s_sum2 == 0 or p_sum2 == 0: return 0.
    
    ratio_1 = s_sum1 / s_sum2
    ratio_2 = p_sum1 / p_sum2
    
    dif = abs(ratio_1 - ratio_2) / ratio_2

    return dif

def barium(library: Library, spectrum: Spectrum) -> bool:
    """
    Determines whether a star exhibits Barium enhancement based on spectral analysis.

    The function analyzes the spectral lines of F and G-type stars to identify potential Barium (Ba) enhancement. In earlier spectral types (earlier than G5), the function primarily checks for enhancement in the Strontium (Sr II) line at 4077 Å, which tends to appear stronger in the presence of Barium enhancement. For later spectral types (later than G5), the function directly examines the Ba II line.

    Parameters
    ----------
    library : Library
        An object representing the spectral library that provides the reference spectra for comparison.
        
    spectrum : Spectrum
        An object representing the observed spectrum of the star.

    Returns
    -------
    Ba : bool
        Returns `True` if Barium enhancement is detected, otherwise `False`.
    """
    
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = spectrum.spt
    spectrum_.lum_code = spectrum_.lum = spectrum.lum
    wave_temp, flux_temp = sp2class(library, spectrum_)

    def calculate_dif(wave_ranges):
        mask = (wave_temp >= wave_ranges[0][0]) & (wave_temp <= wave_ranges[0][1])
        s_sum2, p_sum2 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
        
        mask = (wave_temp >= wave_ranges[1][0]) & (wave_temp <= wave_ranges[1][1])
        s_sum1, p_sum1 = np.sum(flux_temp[mask]), np.sum(spectrum.flux_out[mask])
        
        if len(wave_ranges) > 2:
            mask = (wave_temp >= wave_ranges[2][0]) & (wave_temp <= wave_ranges[2][1])
            s_sum2 += np.sum(flux_temp[mask])
            p_sum2 += np.sum(spectrum.flux_out[mask])
            
            s_sum2 /= 2
            p_sum2 /= 2
        
        ratio_1 = s_sum1 / s_sum2
        ratio_2 = p_sum1 / p_sum2
        
        dif = (ratio_1 - ratio_2) / ratio_1
        return dif

    # Check if the spcode is less than or equal to 32.0 (earlier than G5 stars)
    Ba = False
    if spectrum.spt <= 32.0:
        dif = calculate_dif([(4068.0, 4074.5), (4074.5, 4081.0), (4081.0, 4087.5)])
        
        logging.info(f"Barium function: Sr II: dif = {dif:.6f}")

        if dif >= 0.05:
            Ba = True
    else:
        # Calculate the ratio for Ba II lines (for later than G5 stars)
        dif = calculate_dif([(4520.0, 4539.0), (4546.0, 4560.0)])
        
        logging.info(f"Barium function: Ba index: {dif:.6f}")

        if dif >= 0.05:
            Ba = True

    return Ba

def CH_band2(sp_code: float, lum_code: float, library: Library, spectrum: Spectrum) -> float:
    """
    Computes the ratio of G-band indices between a reference spectrum and an input spectrum 
    based on the given spectral type (`sp_code`) and luminosity class (`lum_code`).

    Parameters
    ----------
    sp_code : float
        The spectral type code, which determines the spectral class of the spectrum.
    
    lum_code : float
        The luminosity class code, which determines the luminosity class of the spectrum.
    
    library : Library
        An instance of the `Library` class containing the spectral data and relevant methods for spectrum generation.
    
    spectrum : Spectrum
        An instance of the `Spectrum` class containing the wavelength and flux data of the input spectrum to be analyzed.
    
    Returns
    -------
    ratio : float
        The ratio of the G-band index of the input spectrum to the G-band index of the synthetic reference spectrum. A positive value indicates a difference in the strength of the G-band absorption between the two spectra.
    """
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.sp_code = spectrum_.spt = sp_code
    spectrum_.lum_code = spectrum_.lum = lum_code
    
    wave_temp, flux_temp = sp2class(library, spectrum_)

    def find_continuum_points(wave, flux, lower_bound, upper_bound):
        mask = (wave >= lower_bound) & (wave <= upper_bound)
        max_flux = np.max(flux[mask])
        max_pos = wave[mask][np.argmax(flux[mask])]
        return max_flux, max_pos

    # Find continuum points for reference spectrum
    cont1, pos1 = find_continuum_points(wave_temp, flux_temp, 4261, 4268)
    cont2, pos2 = find_continuum_points(wave_temp, flux_temp, 4316, 4320)

    # Calculate G-band index for reference spectrum
    mask = (wave_temp >= 4298) & (wave_temp <= 4309)
    cont = cont1 + (cont2 - cont1) * (wave_temp[mask] - pos1) / (pos2 - pos1)
    G1 = np.sum(1.0 - flux_temp[mask] / cont)

    # Find continuum points for input spectrum
    cont1, pos1 = find_continuum_points(spectrum_.wave_out, spectrum_.flux_out, 4261, 4268)
    cont2, pos2 = find_continuum_points(spectrum_.wave_out, spectrum_.flux_out, 4316, 4320)

    # Calculate G-band index for input spectrum
    mask = (spectrum_.wave_out >= 4298) & (spectrum_.wave_out <= 4309)
    cont = cont1 + (cont2 - cont1) * (spectrum_.wave_out[mask] - pos1) / (pos2 - pos1)
    G2 = np.sum(1.0 - spectrum_.flux_out[mask] / cont)

    # Calculate the ratio
    ratio = (G2 - G1) / G1

    return ratio

def lam_boo(library: Library, spectrum: Spectrum) -> int:
    """
    Attempts to classify a star as a Lambda Boo star by comparing the hydrogen and metallic line profiles.

    Parameters
    ----------
    library : Library
        An object containing the library of standard spectra and relevant parameters.
        
    spectrum : Spectrum
        An object representing the input spectrum.

    Returns
    -------
    LB : int
        A flag representing whether the star is classified as a Lambda Boo star.
    
    Notes
    -----
    The function modifies some of the `spectrum` attributes that represent metallicity in-place.
    """
    # Assume the star is a dwarf (luminosity class V)
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.lum_code = spectrum_.lum = 5.0
    logging.info("Classifying this star as a Lambda Boo star")

    # Start H-line type out as F0
    spt_h = 23.0
    
    # Find the best matching spectral type for hydrogen lines using Brent's method
    spt_h = brent(spt_h - 1, spt_h, hydrogen_profile, args=(library, spectrum_))
    spectrum.h_type = spt_h
    logging.info(f"Hydrogen-line type = {spectrum.h_type:.6f}")

    # Find the best matching spectral type for metallic lines using Brent's method
    spt_m = brent(spectrum_.spt - 1, spectrum_.spt, hydrogen_profile, args=(library, spectrum_))
    spectrum.metal_type = spt_m
    logging.info(f"Metallic-line type = {spt_m:.6f}")

    # Decision logic for classifying as Lambda Boo star
    """
    If the hydrogen line type is much later than F0, it probably is not a Lambda Boo star, but a metal-weak FBS, although this point is debateable.
    
    NOTE: If the hydrogen-line type is too early, then determining the metallic-line type is too difficult. So, we are relying here for the classification solely on the weakness of the Mg II line.
    """
    LB = 0
    if (spt_h - spt_m > 2.0 and 17.5 <= spt_h <= 23.5) or (spt_h < 17.5):
        LB = 1

    return LB

def lam_boo_2(library: Library, spectrum: Spectrum) -> int:
    """
    Classifies whether an A-type star is metal-weak and attempts to distinguish between a Lambda Boo star and another type of metal-weak A-type star, such as a horizontal-branch star.

    Parameters
    ----------
    library : Library
        An object containing the library of standard spectra and relevant parameters.
        
    spectrum : Spectrum
        An object representing the input spectrum.

    Returns
    -------
    LB : int
        Classification result:
        - 0 - Not a Lambda Boo star.
        - 1 - Lambda Boo star.
        - 2 - Metal-weak A-type star, possibly a horizontal-branch star.
        - 3 - High vsini star or another non-Lambda Boo star.
    """
    # Exclude stars that are too luminous from being classified as Lambda Boo
    if spectrum.lum < 3.0: return 0

    spectrum_ = copy.deepcopy(spectrum)
    """
    We assume a preliminary luminosity type of IV-V, which is a compromise between the dwarf status of Lambda Boo stars and IV or III type of HB stars. We use that preliminary luminosity type to find the spectral type that best matches the hydrogen lines, and then we iterate further.
    """
    spectrum_.lum_code = spectrum_.lum = 4.5  # Preliminary luminosity type (IV-V)
    logging.info(f"Classifying this star as a metal-weak A-type star. spt = {spectrum.spt:.6f}")
    
    """
    Start with F0 for hydrogen-line type, so that the routine has to work back to the hydrogen maximum, rather than starting near the hydrogen maximum and spuriously moving into the B-type stars.
    """
    spt_h = 23.0

    # Determine hydrogen-line type
    spt_h = brent(spt_h - 1, spt_h, hydrogen_profile, args=(library, spectrum_))
    
    logging.info(f"Hydrogen-line type = {spt_h:.6f}")
    spectrum.h_type = spectrum.sp_code = spt_h

    # If the hydrogen-line type is earlier than F0, we iterate on the luminosity type and the hydrogen-line type
    if spt_h < 23.0:
        spectrum.lum = brent(spectrum_.lum_code - 0.5, spectrum_.lum_code, lum_ratio_min, args=(library, spectrum_))
        spectrum.lum_code = spectrum.lum = min(spectrum.lum, 5.2)
        logging.info(f"Luminosity type = {spectrum.lum:.6f}")

        spt_h = brent(spt_h - 1, spt_h, hydrogen_profile, args=(library, spectrum_))
        logging.info(f"Hydrogen-line type = {spt_h:.6f}")
        spectrum.h_type = spectrum.sp_code = spt_h
        

    # Determine K-line and metallic-line types
    spt_K = brent(16, 17, spt_CaK, args=(library, spectrum_))
    logging.info(f"K-line type = {spt_K:.6f}")
    
    spt_m = brent(16, 17, spt_metal, args=(library, spectrum_))
    logging.info(f"Metallic-line type = {spt_m:.6f}")

    """
    If the dispersion is low, the metallic-line type can be faulty, especially in metal-weak stars. So, if spt_m < spt_K, then correct spt_m to spt_K
    """
    # Correct metallic-line type if necessary
    if spt_m < spt_K - 0.5:
        spt_m = spt_K
        logging.info("Correcting metallic-line type to K-line type")

    spectrum.metal_type = spt_m

    # Classification logic
    """
    If the hydrogen line type is much later than F0, it probably is not a Lambda Boo star, but a metal-weak FBS, although this point is debateable. We limit LB classifications to stars with H - M spectral type differences greater than 2.5, to keep high vsini stars from being spuriously classified as LBs.
    """
    LB = 0
    if spt_h - spt_m >= 2.5 and 17.5 <= spt_h <= 23.5: LB = 1
    if spt_h - spt_m < 2.5: LB = 3  # Likely high vsini star
    """
    NOTE: If the hydrogen-line type is too early, then determining the metallic-line type is too difficult. So, we are relying here for the classification solely on the weakness of the Mg II line. We might add in a criterion based on the K-line??
    """
    if spt_h < 17.5: LB = 1

    # Luminosity-based differentiation
    """
    We use the luminosity type to differentiate between Lambda Boo and FHB stars. For stars earlier than A5, we use the actual luminosity type, determined above.  For stars later than A5, however, we use the ratio between Ca I 4226 and Fe II 4233.
    """
    if spt_h < 20.0 and spectrum.lum < 4.3 and LB == 1: LB = 2
    elif spt_h >= 20.0:
        ratio = ratio_CaI_FeII(spectrum.wave_out, spectrum.flux_out)
        logging.info(f"CaIFeII ratio = {ratio:.6f}")
        if ratio < 1.3: LB = 2

    # Final check on metal-hydrogen difference
    if abs(spt_h - spt_m) < 2.5: LB = 3

    return LB

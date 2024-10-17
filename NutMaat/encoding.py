import logging
import numpy as np

from .library import *
from .spectrum import *
from .evaluate import *

def sub_class(spt: float) -> float:
    """
    Computes the subclass value for a given spectral code.

    Parameters
    ----------
    spt : float
        The spectral code value.

    Returns
    -------
    sc : float
        The corresponding subclass value.
    """
    sp_codes = np.array([20.0, 21.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0])

    subs = np.array([0.0, 2.0, 5.0, 7.0, 8.0, 10.0, 11.0, 13.0, 14.0, 15.0, 17.0, 20.0, 23.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 35.5])

    # Find the index `k` where sp falls between spcode[k] and spcode[k+1]
    for i in range(len(sp_codes) - 1):
        if spt >= sp_codes[i] and spt < sp_codes[i + 1]:
            # Perform linear interpolation
            sc = subs[i] + (subs[i + 1] - subs[i]) * (spt - sp_codes[i]) / (sp_codes[i + 1] - sp_codes[i])
            break

    # If no match is found, return the nearest bound based on the input value
    if spt >= sp_codes[-1]:
        sc = subs[-1]
    elif spt < sp_codes[0]:
        sc = subs[0]
    
    return sc

def spt_code(spt: str) -> float:
    """
    Returns a rough running number equivalent of a temperature type.

    This function maps the first character of a spectral type string to a corresponding 
    numerical code based on predefined spectral types. If the character does not match 
    any of the predefined types, the function returns -10.0.

    Parameters
    ----------
    spt : str
        The spectral type string. The function only considers the first character of this string.

    Returns
    -------
    code : float
        The numerical code corresponding to the spectral type, or -10.0 if the type is not recognized.
    """
    if spt == "": return -10
    
    spectral_type_to_code = {
        'O': 3.0,
        'B': 12.0,
        'A': 20.0,
        'F': 26.0,
        'G': 32.0,
        'K': 36.0,
        'M': 43.0
    }
    # Return the code for the spectral type, or -10.0 if not found
    return spectral_type_to_code.get(spt[0], -10.0)

def spt2code(spt: str) -> float:
    """
    Converts a spectral type string to a corresponding numeric code.
    
    Parameters
    ----------
    spt : str
        Spectral type string (e.g., 'O3', 'B0', 'A5', etc.)
    
    Returns
    -------
    code : float
        Numeric code corresponding to the spectral type, or -10.0 if the type is not recognized.
    """
    code_dict = {
        "O3": 0.0, 
        "O4": 1.0, 
        "O5": 2.0, 
        "O6": 3.0, 
        "O7": 4.0, 
        "O8": 5.0, 
        "O9": 6.0,
        "B0": 7.0, 
        "B1": 8.0, 
        "B2": 9.0, 
        "B3": 10.0, 
        "B3+": 10.2, 
        "B4": 11.0, 
        "B5": 12.0, 
        "B6": 12.5, 
        "B7": 13.0, 
        "B8": 14.0, 
        "B9": 15.0,
        "A0": 16.0, 
        "A1": 17.0, 
        "A1.5": 17.5, 
        "A2": 18.0, 
        "A3": 19.0, 
        "A4": 19.5, 
        "A5": 20.0, 
        "A6": 20.5, 
        "A7": 21.0, 
        "A8": 21.7, 
        "A9": 22.5,
        "F0": 23.0, 
        "F1": 23.5, 
        "F2": 24.0, 
        "F3": 25.0, 
        "F4": 25.5, 
        "F5": 26.0, 
        "F5.5": 26.5, 
        "F6": 27.0, 
        "F7": 27.5, 
        "F8": 28.0, 
        "F8.5": 28.5, 
        "F9": 29.0, 
        "F9.5": 29.5,
        "G0": 30.0, 
        "G0+": 30.1, 
        "G0.5": 30.25, 
        "G0-": 29.8, 
        "G1": 30.5, 
        "G1.5": 30.75, 
        "G2": 31.0, 
        "G2-": 30.9, 
        "G2+": 31.15, 
        "G2.5": 31.2, 
        "G3": 31.3, 
        "G3+": 31.4, 
        "G4": 31.7, 
        "G4.5": 31.85, 
        "G5": 32.0, 
        "G5+": 32.1, 
        "G5.5": 32.15, 
        "G6": 32.3, 
        "G6.5": 32.5, 
        "G7": 32.7, 
        "G8": 33.0, 
        "G8+": 33.1, 
        "G9": 33.5, 
        "G9+": 33.6, 
        "G9-": 33.35,
        "K0": 34.0, 
        "K0-": 33.9, 
        "K1": 35.0, 
        "K2": 36.0, 
        "K2.5": 36.5, 
        "K3": 37.0, 
        "K3+": 37.2, 
        "K3-": 36.8, 
        "K3.5": 37.5, 
        "K4": 38.0, 
        "K4-": 37.8, 
        "K4.5": 38.5, 
        "K5": 39.0, 
        "K5-": 38.8, 
        "K5.5": 39.25, 
        "K5+": 39.125, 
        "K6": 39.5, 
        "K6-": 39.375, 
        "K6+": 39.625, 
        "K6.5": 39.75, 
        "K7": 40.0, 
        "K7-": 39.875, 
        "K7+": 40.142, 
        "K8": 40.28, 
        "K9": 40.428,
        "M0": 40.714, 
        "M0+": 40.857, 
        "M0-": 40.571, 
        "M0.5": 41.0, 
        "M1": 41.5, 
        "M1-": 41.3, 
        "M1+": 41.7, 
        "M1.5": 42.0, 
        "M1.5+": 42.2, 
        "M2": 42.5, 
        "M2+": 42.7, 
        "M2-": 42.4, 
        "M2.5": 43.0, 
        "M3": 43.5, 
        "M3.5": 44.0, 
        "M4": 44.5, 
        "M4.5": 45.0, 
        "M5": 45.5, 
        "M6": 46.5, 
        "M7": 47.5, 
        "M8": 48.5, 
        "M9": 49.5
    }

    return code_dict.get(spt, -10.0)

def code2spt(sp_code: float) -> str:
    """
    Converts a spectral code (`sp_code`) into its corresponding spectral type (`SPT`).

    Parameters
    ----------
    sp_code : float
        The spectral code.

    Returns
    -------
    SPT : str
        The spectral type corresponding to `sp_code`.
    """
    codes = [
        -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.3, 12.7, 13.5, 14.5, 15.3, 15.7, 16.5, 17.5, 18.5, 19.3, 19.7, 20.3, 20.7, 21.5, 22.1, 22.75, 23.25, 23.75, 24.5, 25.25, 25.75, 26.5, 27.25, 27.75, 28.6, 29.5, 30.25, 30.75, 31.25, 31.5, 31.75, 32.25, 32.55, 32.85, 33.25, 33.75, 34.5, 35.5, 36.5, 37.5, 38.5, 39.4, 39.6, 40.5, 40.8, 41.2, 41.8, 42.2, 42.8, 43.2, 43.8, 44.2, 44.8, 45.25, 46.0, 47.0, 48.0, 49.0, 50.0
    ]
    
    if sp_code < codes[0] or sp_code > codes[-1]:
        return "??"
    
    types = [
        "O3", "O4", "O5", "O6", "O7", "O8", "O9", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B9.5", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "M0", "M0.5", "M1", "M1.5", "M2", "M2.5", "M3", "M3.5", "M4", "M4.5", "M5", "M6", "M7", "M8", "M9"
    ]
    
    for i in range(len(codes)-1):
        if codes[i] <= sp_code < codes[i+1]:
            return types[i]

    return "??"

def code2lum(lum_code: float) -> str:
    """
    Converts a luminosity code (`lum_code`) into its corresponding luminosity type (`LUM`).

    Parameters
    ----------
    lum_code : float
        The luminosity code.

    Returns
    -------
    LUM : str
        The luminosity type corresponding to `lum_code`.
    """
    codes = [-1.5, -0.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 6.0]
    
    if lum_code > codes[-1] or lum_code < codes[0]:
        return "?"
    
    types = ["0", "Ia", "Iab", "Ib", "Ib-II", "II", "II-III", "III", "III-IV", "IV", "IV-V", "V", "V-"]
    
    for i in range(len(codes) - 1):
        if codes[i] <= lum_code < codes[i+1]:
            return types[i]
    return "?"

def quality(spectrum: Spectrum, chi2: float) -> str:
    """
    Assigns a quality tag based on the chi-squared value and the spectral type.

    Parameters
    ----------
    spectrum : Spectrum
        An object representing the input spectrum.
    
    chi2 : float
        The chi-squared value.

    Returns
    -------
    qual : str
        The quality tag indicating the fit quality, appended with the `note` attribute.
    """

    # Determine quality tag based on spectral type and chi-squared value
    if spectrum.spt >= 39.0:
        if chi2 < 1.0e-4:
            qual = "|  excel  |"
        elif chi2 < 5.0e-3:
            qual = "|  vgood  |"
        elif chi2 < 5.0e-2:
            qual = "|  good   |"
        elif chi2 < 1.0e-1:
            qual = "|  fair   |"
        else:
            qual = "|  poor   |"
    else:
        if chi2 < 1.0e-4:
            qual = "|  excel  |"
        elif chi2 < 1.0e-3:
            qual = "|  vgood  |"
        elif chi2 < 1.0e-2:
            qual = "|  good   |"
        elif chi2 < 5.0e-2:
            qual = "|  fair   |"
        else:
            qual = "|  poor   |"

    # Append note to the quality tag
    qual += spectrum.note

    return qual

def low_SN(flux: np.ndarray) -> int:
    """
    Detects low signal-to-noise (S/N) based on negative flux values.
    
    Parameters
    ----------
    flux : np.ndarray
        Flux array.
    
    Returns
    -------
    S2N : int:
        Returns 0 if S/N is normal, 1 if more than 3 but fewer than 50 flux points are negative, 2 if 50 or more flux points are negative.
    """
    # Count the number of negative flux values
    count = np.sum(flux < 0.0)

    # Determine S/N classification based on the count of negative flux points
    if 3 < count < 50:
        return 1  # Low S/N but classifiable
    elif count >= 50:
        return 2  # Unclassifiable due to very low S/N
    else:
        return 0  # Normal S/N

def emission(wave: np.ndarray, flux: np.ndarray) -> int:
    """
    Detects most emission-line stars and attempts to identify the type.

    Parameters
    ----------
    wave : np.ndarray
        Array of wavelength values.
        
    flux : np.ndarray
        Array of flux values.

    Returns
    -------
    e : int
        Emission type classification.
            - 0 = absorption-line spectrum,
            - 1 = unidentified emission-line spectrum,
            - 2 = WN,
            - 3 = Helium nova,
            - 4 = WC
    """
    
    # Normalize flux by its maximum value
    flux_max = np.max(flux[100:-19])
    wave_max = wave[np.argmax(flux[100:-19]) + 100]
    
    logging.info(f"wave_max = {wave_max:7.2f}")
    
    flux_ = flux / flux_max
    

    # Compute mean and standard deviation
    avg = np.mean(flux_)
    s_dev = np.std(flux_)
    logging.info(f"Emission function: avg = {avg:.6f} s_dev = {s_dev:.6f}")

    # Emission detection criteria
    e = 0
    if ((avg <= 0.27 and s_dev < 0.185185185 * avg + 0.1) or (avg > 0.27 and s_dev < -0.1 * avg + 0.130)): e = 1

    # Check for He II or Carbon emission lines
    if e == 1:
        if 4680.0 < wave_max <= 4690.0: e = 2
        if 4655.0 < wave_max <= 4665.0: e = 4

    # If He II emission lines are present, check for nova or WN
    if e == 2:
        mask_4770 = (wave >= 4770.0) & (wave <= 4780.0)
        ave4770 = np.mean(flux_[mask_4770])
        flux_half = (1.0 + ave4770) / 2.0

        mask_4840_4870 = (wave >= 4840.0) & (wave <= 4870.0) & (flux_ > flux_half)
        if np.any(mask_4840_4870): e = 3

    return e

def detect_NN(library: Library, spectrum: Spectrum) -> int:
    """
    Detects the type of star based on its spectrum.

    Parameters
    ----------
    library : Library object
        An object containing the library of standard spectra and relevant parameters.

    spectrum : Spectrum object
        An object representing the input spectrum. 

    Returns
    -------
    code : int
        Star type classification code.
    """

    # Check for low S/N
    SN = low_SN(spectrum.flux_rebin)
    if SN == 1: return 13
    if SN == 2: return 14

    # Check for emission-line stars
    e = emission(spectrum.wave_rebin, spectrum.flux_rebin)
    logging.info(f"emission e = {e}")

    if e == 1: return 9
    if e == 2: return 10
    if e == 3: return 11
    if e == 4: return 12

    # Determine Hydrogen-line index based on ratio of width to depth
    if library.w_high < 4700.0:
        width, ratio = hyd_rat(spectrum.wave_rebin, spectrum.flux_rebin)
        logging.info(f"NN: ratio = {ratio:.6f} width = {width:.6f}")
        if 0 < ratio < 65 and 2 < width < 27.5: return 0
        if 60 <= ratio <= 125 and 13.0 <= width <= 42: return 13 # in a questionable region
        if width > 26.0 and ratio > 125: return 1 # probably DAs
        if width < 0.0 or ratio < 0.0: return 6
        else: return 6
    else:
        ratio = Carbon(spectrum.wave_rebin, spectrum.flux_rebin)
        logging.info(f"Carbon ratio = {ratio:.6f}")
        if ratio > 3.0: return 8
        
        width, ratio = hyd_rat(spectrum.wave_rebin, spectrum.flux_rebin)
        logging.info(f"NN: ratio = {ratio:.6f} width = {width:.6f}")
        
        # Check for Normal stars #
        if 0 <= ratio <= 65 and 1.9 <= width <= 27.5: return 0
        
        # Check to see if late M-type star #
        index = late_M(spectrum.wave_rebin, spectrum.flux_rebin)
        if (index > 1.0 and 0 < ratio < 20) or index > 2.0: return 0
        
        # Check to see if DA #
        if 60 <= ratio <= 125 and 13.0 <= width <= 42: return 13 # may have defective spectra
        
        # probably DAs
        if width > 26.0 and ratio > 125:
            logging.info(f"NN: ratio = {ratio:.6f} width = {width:.6f}")
            return 1
        
        if width < 0.0 or ratio < 0.0: return 6
        
        # Check for other types of WD and Carbon stars #
        index = DB(spectrum.wave_rebin, spectrum.flux_rebin)
        logging.info(f"DB = {index:.6f}")
        if index > 1.20: return 2
        
        index, C_ratio = DZ(spectrum.wave_rebin, spectrum.flux_rebin)
        logging.info(f"DZ index = {index:.6f} C_ratio = {C_ratio:.6f}")
        
        # This distinguishes DZ stars and CN-strong stars that were not caught with the Carbon function above #
        if index > 1.05 and C_ratio < 1.3: return 4
        else: return 0
        
        # index = DO(spectrum.wave_rebin, spectrum.flux_rebin)
        # if index > 1.02: return 3
    
    # return 6

import copy
import numpy as np

from scipy.constants import speed_of_light
from scipy.optimize import minimize_scalar
from scipy.linalg import solve

from .library import *
from .spectrum import *

def rebin(wave: np.ndarray, flux: np.ndarray, start: float, end: float, width: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Rebins the spectrum.

    Parameters:
    -----------
    wave : numpy.ndarray
        Original wavelength array.
        
    flux : numpy.ndarray
        Original flux array.
        
    start : float
        Start of the new wavelength grid.
        
    end : float
        End of the new wavelength grid.
        
    width : float
        Wavelength step size.

    Returns:
    --------
    wave_new : numpy.ndarray
        Rebinned wavelength array.
        
    flux_new : numpy.ndarray
        Rebinned flux array.
        
    ind_start : int
        Index in the new grid corresponding to the start of the original data.
        
    ind_end : int
        Index in the new grid corresponding to the end of the original data.
    """
    # Create the new wavelength grid
    wave_new = np.arange(start, end, width)
    ind_end = len(wave_new) - 1

    # Determine the start index ind_start in the new grid
    if start > wave[0]:
        ind_start = 0
    else:
        ind_start = int(np.ceil((wave[0] - start) / width))

    # Adjust the end index ind_end based on the original data
    while end > wave[-1]:
        end -= width
        ind_end -= 1

    # Rebin the spectrum
    flux_new = np.zeros_like(wave_new)
    index = 1
    wl = start + ind_start * width

    for i in range(ind_start, ind_end + 1):
        while index < len(wave) and not (wave[index-1] <= wl <= wave[index]):
            index += 1

        if index < len(wave):
            flux_new[i] = flux[index-1] + (flux[index] - flux[index-1]) * (wl - wave[index-1]) / (wave[index] - wave[index-1])

        wl += width

    return wave_new, flux_new, ind_start, ind_end

def vel_shift(library: Library, spectrum: Spectrum, v_step = 2.) -> float:
    """
    Calculates the velocity shift between an observed spectrum and a template spectrum
    from a spectral library by minimizing the chi-squared difference between the 
    two spectra.

    Parameters
    ----------
    library : Library
        An object representing the spectral library.
    
    spectrum : Spectrum
        An object representing the observed spectrum.
    
    v_step : float, optional
        The step size in km/s for the initial velocity search grid. Default is 2.0 km/s.

    Returns
    -------
    vel : float
        The velocity shift in km/s that best aligns the observed spectrum with the 
        template spectrum.
    """
    c = speed_of_light/1e3  # km/s
    
    # Load the spectra #
    w_tmplt, f_tmplt = get_spectrum(os.path.join(library.lib_path, spectrum.irbn))
    w_obs, f_obs = spectrum.wave, spectrum.flux
    
    # Rebin the spectra #
    w_trbn, f_trbn, _, _ = rebin(w_tmplt, f_tmplt, library.w_low, library.w_high, 0.01)
    w_orbn, f_orbn, _, _ = rebin(w_obs, f_obs, library.w_low, library.w_high, 0.01)
    
    # Determine rebinning limits #
    N = len(w_orbn)
    m, n = 0, N - 1
    for i in range(N):
        if abs(w_trbn[i] - (library.w_low + 7.)) < 0.005: m = i
        if abs(w_trbn[i] - (library.w_high - 7.)) < 0.005: n = i

    # Determine the central wavelength and velocity bounds #
    w_center = (library.w_high + library.w_low) / 2.0
    V = -12.0 * c / w_center
    V_end = 12.0 * c / w_center
    
    def chi_squared(V):
        d_wav = w_trbn * V / c
        w_trbn2 = w_trbn + d_wav
        f_trbn2 = copy.deepcopy(f_trbn)
        _, f_trbn3, _, _ = rebin(w_trbn2, f_trbn2, library.w_low, library.w_high, 0.01)
        mask = (f_trbn3[m:n+1] != 0.0)
        chi2 = np.mean((f_trbn3[m:n+1][mask] - f_orbn[m:n+1][mask]) ** 2)
        return chi2

    # Minimize the chi-squared value over the velocity range #
    res = minimize_scalar(chi_squared, bounds=(V, V_end), method='bounded', options={'xatol': v_step})
    v_min = res.x

    # Fine-tune the velocity shift using a parabolic fit #
    v_values = np.array([v_min - v_step, v_min, v_min + v_step])
    chi_values = np.array([chi_squared(v) for v in v_values])

    A = np.vstack([v_values**2, v_values, np.ones_like(v_values)]).T
    coeffs = solve(A, chi_values)
    
    vel = -coeffs[1] / (2.0 * coeffs[0])

    return vel

def mk_prelim(library: Library, spectrum: Spectrum) -> None:
    """
    Processes an input spectrum in place by normalizing it, potentially applying a velocity shift,
    and rebinning to a new wavelength grid.
    
    Parameters:
    -----------
    library : Library object
        An object representing the spectral library.
    
    spectrum : Spectrum object
        An object representing the observed spectrum.
    """
    c = speed_of_light / 1e3  # km/s

    # Check flags
    flag_shift = 'shift' in library.prelim
    flag_norm = 'norm' in library.prelim

    # Load the input spectrum
    wave, flux = spectrum.wave.copy(), spectrum.flux.copy()

    # Normalize the spectrum if required
    if flag_norm:
        mask = (wave > 4490.0) & (wave < 4520.0)
        flux_max = np.max(flux[mask]) if np.any(mask) else 1.0
        flux /= flux_max
    
    # Get the current working directory
    current_dir = os.getcwd()
    # Define the output file name and create the full file path
    nor_path = os.path.join(current_dir, "normal.dat")
    # Open the file in write mode and write the data from wavein and flux
    with open(nor_path, 'w') as nor:
        for i in range(len(wave)):
            nor.write(f"{wave[i]:.6f} {flux[i]:.6f}\n")

    spectrum.wave, spectrum.flux = wave, flux
    
    # Apply velocity shift if required
    library_ = copy.deepcopy(library)
    library_.w_low += 100
    library_.w_high -= 100
    if flag_shift:
        vel = vel_shift(library_, spectrum)
        wave /= (1.0 + vel / c)

    # Rebin the spectrum
    spectrum.wave_out, spectrum.flux_out, _, _ = rebin(wave, flux, library.w_low, library.w_high, library.space)
    
    # Define the output file name and create the full file path
    temp_path = os.path.join(current_dir, "temp.out")
    # Open the file in write mode and write the data from wavein and flux
    with open(temp_path, 'w') as temp:
        for i in range(len(spectrum.wave_out)):
            temp.write(f"{spectrum.wave_out[i]:.6f} {spectrum.flux_out[i]:.6f}\n")

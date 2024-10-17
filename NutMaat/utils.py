import copy
import math
import numpy as np
import pandas as pd
from .library import *
from .spectrum import *

# Constants (these are the typical values used in optimization problems)
GOLD = 1.618034     # Golden ratio constant
GLIMIT = 100.0      # Maximum magnification allowed for parabolic fit
TINY = 1e-20        # A small number to avoid division by zero
ITMAX = 20000       # Maximum allowed iterations
CGOLD = 0.3819660   # Golden ratio constant (used in Brent's method)
ZEPS = 1.0e-10      # Small value to prevent division by zero

def lib_conform(library: Library, spectrum: Spectrum) -> int:
    """
    Checks if the wavelength limits conform for the given spectrum.

    Parameters
    ----------
    library : Library
        The library to conform to.
        
    spectrum : Spectrum 
        The spectrum to be checked for conformity.

    Returns
    -------
    int:
        - 0 if the spectrum conforms,
        - 1 if the lower wavelength limit exceeds the library's lower limit by more than 100.0,
        - 2 if the upper wavelength limit falls below the library's upper limit by more than 100.0.
    """
    # Currently, this function only checks to see if the wavelength limits conform
    if spectrum.wave[0] > library.w_low + 100.0:
        return 1
    if spectrum.wave[-1] < library.w_high - 100.0:
        return 2

    return 0

def match(library: Library, spectrum: Spectrum) -> float:
    """
    Matches the program spectrum with the library spectrum and computes the chi-squared value.

    Parameters
    ----------
    library : Library
        An instance of the Library class containing the synthetic spectrum data.
        
    spectrum : Spectrum
        An instance of the Spectrum class containing observed spectrum data.

    Returns
    -------
    chi2 : float
        The chi-squared value normalized by the number of data points used in the calculation.
        If a match file path is provided in `spectrum.lib_match`, the matched synthetic
        spectrum is written to this file.
    """
    # Generate the synthetic spectrum for the given spectral type and luminosity
    wave_temp, flux_temp = sp2class(library, spectrum)

    # Open the library match file and save the matched data
    if spectrum.lib_match:
        with open(spectrum.lib_match, "w") as mch:
            for x, y in zip(wave_temp, flux_temp):
                mch.write(f"{x} {y}\n")

    # Determine the chi-squared sum based on the spectral type
    if spectrum.spt < 41.0:
        mask = (wave_temp >= library.w_low + 100.0) & (wave_temp <= library.w_high - 100.0)
    else:
        mask = ((wave_temp >= library.w_low + 100.0) & (wave_temp >= 4400.0)) & (wave_temp <= library.w_high - 100.0)

    # Compute the normalized chi-squared value
    chi2 = np.mean((flux_temp[mask] - spectrum.flux_out[mask]) ** 2)

    return chi2

def lst_sqr(wave: np.ndarray, flux: np.ndarray, w_low: float, w_high: float) -> tuple[float, float]:
    """
    Performs a linear least squares fit to a specified range of wavelength values, returning the coefficients of the best-fit line. The fitting is done using NumPy's `polyfit` function.

    Parameters
    ----------
    wave : np.ndarray
        A 1D array containing the wavelength values of the spectrum.
    
    flux : np.ndarray
        A 1D array containing the flux values corresponding to the wavelengths in `wave`.
    
    w_low : float
        The lower bound of the wavelength range over which to perform the linear fit.
    
    w_high : float
        The upper bound of the wavelength range over which to perform the linear fit.

    Returns
    -------
    a, b : tuple[float, float]
        The coefficients `a` (intercept) and `b` (slope) of the linear fit, where the fit is modeled as `flux = a + b * wave`.

    Notes
    -----
    - If the specified wavelength range does not contain any data points, the function returns `(0.0, 0.0)`.
    """
    # Mask to select X values within the specified range
    mask = (wave >= w_low) & (wave <= w_high)
    x = wave[mask]
    y = wave[mask]

    if len(x) == 0: return 0.0, 0.0
    
    # Perform linear least squares fitting using numpy's polyfit
    b, a = np.polyfit(x, y, 1)

    return a, b

def find_best(Iter: list[Results], NI: int) -> int:
    """
    Finds the index of the result with the minimum chi-squared value in the first `NI` elements of the list.

    Parameters
    ----------
    Iter : list
        A list where each element is a `Results` object with at least a `chi2` attribute.
        
    NI : int
        The number of items to consider from the list, starting from the first element.

    Returns
    -------
    best_index : int
        The index of the element with the minimum chi-squared value among the first `NI` elements.
    """
    chi_min = float('inf')
    best_index = NI

    for i in range(1, NI+1):
        if Iter[i].chi2 <= chi_min:
            chi_min = Iter[i].chi2
            best_index = i

    return best_index

def sp2class(library: Library, spectrum: Spectrum) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates spectral fluxes from a library of spectra based on the provided spectral 
    type and luminosity class codes using linear interpolation.

    Parameters
    ----------
    library : Library 
        An object representing the library containing the spectra files. 
    
    spectrum : Spectrum 
        An object representing the spectrum to be interpolated, including spectral type and luminosity class.

    Returns
    -------
    wave_temp : np.ndarray
        An array of wavelengths corresponding to the interpolated flux array.
        
    flux_temp : np.ndarray
        An array of interpolated flux values based on the given spectral type and luminosity class codes.

    Notes
    -----
    - The function handles extrapolation by clamping values when the provided spectral or 
      luminosity class exceeds the range covered by the library.
    - The interpolation is performed across four surrounding spectra, and linear interpolation 
      is applied based on the fractional difference from the nearest grid points.
    """
    # Define the fixed arrays
    t = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 36.0, 37.0, 39.0, 40.0, 40.7, 42.5, 44.0, 45.5])
    l = np.array([1.0, 3.0, 5.0])
    
    I, J = len(t), len(l)

    # Find the indices for spectral and luminosity codes #
    k2 = np.searchsorted(t, spectrum.spt, side='right')
    k1 = k2 - 1
    l2 = np.searchsorted(l, spectrum.lum, side='right')
    l1 = l2 - 1
    
    # Handle extrapolation beyond the limits #
    if k1 < 0:
        k1, k2 = 0, 1
    elif k2 >= I:
        k1, k2 = I - 2, I - 1
    if l1 < 0:
        l1, l2 = 0, 1
    elif l2 >= J:
        l1, l2 = J - 2, J - 1

    # Compute interpolation parameters
    T = (spectrum.spt - t[k1]) / (t[k2] - t[k1])
    U = (spectrum.lum - l[l1]) / (l[l2] - l[l1])
    
    paths = [
        os.path.join(library.lib_path, f't{int(t[k1] * 10):03d}l{int(l[l1] * 10):02d}p00.rbn'),
        os.path.join(library.lib_path, f't{int(t[k2] * 10):03d}l{int(l[l1] * 10):02d}p00.rbn'),
        os.path.join(library.lib_path, f't{int(t[k2] * 10):03d}l{int(l[l2] * 10):02d}p00.rbn'),
        os.path.join(library.lib_path, f't{int(t[k1] * 10):03d}l{int(l[l2] * 10):02d}p00.rbn')
    ]
    
    fluxes = [0]
    wave_temp, fluxes[0] = get_spectrum(paths[0])
    for path in paths[1:]:
        fluxes.append(get_spectrum(path)[1])
    
    # Interpolate the spectra
    flux_temp = (1.0 - T) * (1.0 - U) * fluxes[0] + \
        T * (1.0 - U) * fluxes[1] + \
            T * U * fluxes[2] + \
                (1.0 - T) * U * fluxes[3]
    
    return wave_temp, flux_temp

def spt2min(pcode: list[float], library: Library, spectrum: Spectrum) -> float:
    """
    Calculates the minimized chi-squared value for a given spectral type and luminosity class.

    Parameters
    ----------
    pcode : list[float]
        A list containing two elements: the spectral type code and the luminosity class code.
        
    library : Library
        An object representing the library containing the spectra files. 
    
    spectrum : Spectrum
        An object representing the spectrum for which the chi-squared value is calculated.
    
    Returns
    -------
    chi2 : float
        The minimized chi-squared value between the observed and interpolated flux values, 
        normalized by the number of data points within the specified wavelength range.
    
    Notes
    -----
    - The function checks whether the spectral type code is within the valid range defined 
      by `library.s_cool` and `library.s_hot`.
    - If the spectral code is outside this range, a large chi-squared value (1e5) is returned.
    - The chi-squared value is normalized by the width of the valid wavelength range.
    """
    # Extract spectral and luminosity codes
    sp_code = pcode[0]
    lum_code = pcode[1]

    # Check if the spectral code is within the valid range
    if sp_code >= library.s_cool or sp_code <= library.s_hot:
        return 1e5
    
    spectrum_ = copy.deepcopy(spectrum)
    spectrum_.spt = spectrum_.sp_code = sp_code
    spectrum_.lum = spectrum_.lum_code = lum_code
    # Interpolate the spectral data
    wave, flux = sp2class(library, spectrum_)

    # Calculate chi-squared in the specified wavelength range
    mask = (wave >= library.w_low + 100.0) & (wave <= library.w_high - 100.0)
    chi2 = np.sum((flux[mask] - spectrum_.flux_out[mask])**2)

    # Normalize chi-squared by the wavelength range
    chi2 /= (library.w_high - library.w_low - 200.0)

    return chi2

def f_flux(wave_in: np.ndarray, flux_in: np.ndarray, band: float) -> float:
    """
    Computes the flux-integral over a specified wavelength band using linear interpolation of 
    a transmission function.

    Parameters
    ----------
    wave_in : np.ndarray
        Array of input wavelengths for the observed spectrum.
        
    flux_in : np.ndarray
        Array of flux values corresponding to the wavelengths in `wave_in`.
        
    band : float
        Wavelength shift to be applied to the transmission function.

    Returns
    -------
    flux_int : float
        The flux-integral, normalized by the integral of the transmission function over the 
        wavelength band.

    Notes
    -----
    - The transmission function is pre-defined and symmetric, ranging from -10 to 10 units in 
      wavelength and with a peak value of 1.0 in the center.
    - The trapezoidal rule is used to approximate the flux-integral and the integral of the 
      transmission function for normalization.
    - The wavelength range of the input spectrum should overlap with the shifted transmission 
      function range for meaningful results.
    """
    # Define the wavelength and transmission arrays
    wave = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    trans = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    # Shift the wavelength array
    wave_ = wave + band

    # Initialize variables
    integral = 0.0
    A = 0.0
    a1 = 0.0
    y1 = 0.0

    j, k = 0, 1
    # Iterate over the wavelength points
    while j < len(wave_in) and wave_in[j] < wave_[-1]:
        if wave_in[j] <= wave_[0]:
            j += 1
            continue
        if wave_in[j] >= wave_[-1]:
            break

        # Find the interval in the shifted wavelength array
        while k < len(wave_):
            if wave_in[j] > wave_[k - 1] and wave_in[j] <= wave_[k]:
                break
            k += 1

        # Linear interpolation of the transmission function
        ft = trans[k - 1] + (trans[k] - trans[k - 1]) * (wave_in[j] - wave_[k - 1]) / (wave_[k] - wave_[k - 1])
        y2 = ft * flux_in[j]
        a2 = ft

        # Trapezoidal rule for integration
        integral += 0.5 * (y1 + y2) * (wave_in[j] - wave_in[j - 1])
        A += 0.5 * (a1 + a2) * (wave_in[j] - wave_in[j - 1])

        a1 = a2
        y1 = y2
        j += 1

    return integral / A

def template_DSO(wave_o: np.ndarray, flux_o: np.ndarray, wave_t: np.ndarray, flux_t: np.ndarray) -> np.ndarray:
    """
    Adjusts the flux values of the input (observed) spectrum by applying a template scaling factor derived 
    from a reference (template) spectrum. The adjustment is done by calculating flux ratios over specific 
    wavelength bands, which are then used to interpolate the ratio across the entire wavelength range of 
    the input spectrum.

    Parameters
    ----------
    wave_o : np.ndarray
        Wavelength array of the observed spectrum.
        
    flux_o : np.ndarray
        Flux array corresponding to `wave_o` for the observed spectrum.
        
    wave_t : np.ndarray
        Wavelength array of the reference (template) spectrum.
        
    flux_t : np.ndarray
        Flux array corresponding to `wave_t` for the template spectrum.
    
    Returns
    -------
    flux_new : np.ndarray
        The flux values of the observed spectrum after adjustment using the template scaling factor.

    Notes
    -----
    - The function uses predefined wavelength bands to compute flux ratios between the observed and 
      template spectra.
    - The flux ratios are interpolated between the bands to create a smooth scaling function across 
      the entire wavelength range of the observed spectrum.
    - A normalization step is applied at the end, which scales the adjusted fluxes so that their 
      maximum value within the 4502–4508 Å region is 1.0, ensuring consistency in the comparison.
    """
    # Define the wavelength bands
    bands = np.array([3875.0, 4020.0, 4211.0, 4500.0, 4570.0, 4805.0, 4940.0, 5100.0, 5450.0])
    
    # Allocate arrays for flux calculations
    flux_new = np.zeros_like(flux_o)
    flux_in = np.zeros_like(bands)
    flux_temp = np.zeros_like(bands)
    ratio = np.zeros_like(bands)

    # Find the maximum relevant index in the bands array
    n_max = np.searchsorted(bands + 10.0, wave_o[-1], side='right')

    # Calculate flux ratios
    for i in range(n_max):
        flux_in[i] = f_flux(wave_o, flux_o, bands[i])
        flux_temp[i] = f_flux(wave_t, flux_t, bands[i])
        ratio[i] = flux_temp[i] / flux_in[i]

    # Interpolate ratios and adjust flux values
    for i in range(len(wave_o)):
        if wave_o[i] < bands[0]:
            rat = ratio[0] + (ratio[1] - ratio[0]) * (wave_o[i] - bands[0]) / (bands[1] - bands[0])
        elif wave_o[i] >= bands[n_max - 1]:
            rat = ratio[n_max - 2] + (ratio[n_max - 1] - ratio[n_max - 2]) * (wave_o[i] - bands[n_max - 2]) / (bands[n_max - 1] - bands[n_max - 2])
        else:
            k = np.searchsorted(bands, wave_o[i], side='right') - 1
            rat = ratio[k] + (ratio[k + 1] - ratio[k]) * (wave_o[i] - bands[k]) / (bands[k + 1] - bands[k])
        
        flux_new[i] = rat * flux_o[i]

    # Normalize flux values
    flux_max = np.max(flux_new[(wave_o >= 4502.0) & (wave_o <= 4508.0)])
    if flux_max > 0:
        flux_new /= flux_max
    
    return flux_new

def format_df(spectrum: Spectrum) -> pd.DataFrame:
    """
    Creates a DataFrame row containing specific attributes of the given `Spectrum` object. 
    The DataFrame includes spectral type (SPT), luminosity class (LUM), quality indicator, 
    and additional notes, along with the chi-squared value corresponding to the best fit 
    iteration of the spectral analysis.

    Parameters
    ----------
    spectrum : Spectrum
        The Spectrum object containing attributes such as spectral type (SPT), luminosity 
        class (LUM), quality of the fit (qual), and notes (note). It also includes the 
        chi-squared values for different fits in the `Iter` list, and the total number of 
        iterations (NI).

    Returns
    -------
    df_row : pandas.DataFrame
        A DataFrame containing a single row with the attributes:
        - 'name'   : Name of the spectrum
        - 'SPT'    : Spectral type of the star
        - 'LUM'    : Luminosity class of the star
        - 'quality': Quality of the spectrum fit
        - 'note'   : Additional notes related to the spectrum
        - 'chi2'   : The chi-squared value for the best fit in the iteration list.
    """
    # Extract the required attributes from the Spectrum object
    J = find_best(spectrum.Iter, spectrum.NI)
    chi2 = spectrum.Iter[J].chi2
    data = {
        'name': spectrum.name,
        'SPT': spectrum.SPT,
        'LUM': spectrum.LUM,
        'quality': spectrum.qual,
        'note': spectrum.note,
        'chi2': chi2
    }
    
    # Create a DataFrame with a single row
    df_row = pd.DataFrame([data])
    
    return df_row

def brent(ax: float, bx: float, func: callable, tol: float = 0.001, args: tuple = None) -> float:
    """
    Implements Brent's method to find the local minimum of a unimodal function. 
    This method first brackets the minimum using `mnbrak` and then iteratively 
    narrows the interval containing the minimum until the tolerance `tol` is met.

    Parameters
    ----------
    ax : float
        One endpoint of the initial bracketing interval.
        
    bx : float
        The initial guess for the minimum value, between `ax` and `cx`.
        
    func : callable
        The function to minimize. It must take a float as input and return a float.
        
    tol : float
        The tolerance for convergence (default is 0.001). The algorithm terminates when the 
        interval size is reduced to within `tol`.
        
    args : tuple, optional
        Additional arguments to pass to `func`.

    Returns
    -------
    xmin : float
        The x-value that minimizes `func`.

    Raises
    ------
    RuntimeError
        If the maximum number of iterations (`ITMAX`) is exceeded.
    """
    ax, bx, cx = mnbrak(ax, bx, func, args)
    
    a = min(ax, cx)
    b = max(ax, cx)
    x = w = v = bx
    
    if args != None: fx = fw = fv = func(x, *args)
    else: fx = fw = fv = func(x)
    e = 0.0  # Distance moved on the step before last
    d = 0.0  # Distance for the current step
    
    for _ in range(ITMAX):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + ZEPS  # Tolerance based on current x
        tol2 = 2.0 * tol1

        # Check for convergence
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            return x

        # Parabolic fit if previous step was large enough
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0.0:
                p = -p

            q = abs(q)
            etemp = e
            e = d

            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                # Use golden section step
                e = (a if x >= xm else b) - x
                d = CGOLD * e
            else:
                # Parabolic step
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = math.copysign(tol1, xm - x)
        else:
            # Use golden section step
            e = (a if x >= xm else b) - x
            d = CGOLD * e

        # Make sure the step is larger than the tolerance
        u = x + (d if abs(d) >= tol1 else math.copysign(tol1, d))
        if args != None: fu = func(u, *args)
        else: fu = func(u)

        # Update the points and values
        if fu <= fx:
            if u >= x: a = x
            else: b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x: a = u
            else: b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    raise RuntimeError("Too many iterations in Brent's method")

def mnbrak(ax: float, bx: float, func: callable, args: tuple = None) -> tuple[float, float, float]:
    """
    Brackets a local minimum of a given function using the Golden Section search technique.
    
    Parameters
    ----------
    ax : float
        The initial guess for the first point of the bracket.
        
    bx : float
        The second guess for the bracket.
        
    func : callable
        The objective function to minimize. It should accept a float as input and return a float.
        
    args : tuple, optional
        Additional arguments to pass to `func`. Defaults to None.
    
    Returns
    -------
    ax, bx, cx : tuple
        A tuple `(ax, bx, cx)` representing three points that bracket a local minimum of the function, 
        where the function values at these points follow: `f(ax) > f(bx) < f(cx)`.
    """
    # Initial evaluations of the function at ax and bx
    if args != None:
        fa = func(ax, *args)
        fb = func(bx, *args)
    else:
        fa = func(ax)
        fb = func(bx)

    # Ensure the function value at bx is less than at ax
    if fb > fa:
        ax, bx = bx, ax
        fa, fb = fb, fa

    # First guess for cx
    cx = bx + GOLD * (bx - ax)
    if args != None: fc = func(cx, *args)
    else: fc = func(cx)

    # Start the bracketing process
    while fb > fc:
        r = (bx - ax) * (fb - fc)
        q = (bx - cx) * (fb - fa)
        denom = 2.0 * math.copysign(max(abs(q - r), TINY), q - r)
        u = bx - ((bx - cx) * q - (bx - ax) * r) / denom
        ulim = bx + GLIMIT * (cx - bx)

        if (bx - u) * (u - cx) > 0.0:
            if args != None: fu = func(u, *args)
            else: fu = func(u)
            if fu < fc:
                ax, bx = bx, u
                fa, fb = fb, fu
                return ax, bx, cx
            elif fu > fb:
                cx = u
                fc = fu
                return ax, bx, cx
            u = cx + GOLD * (cx - bx)
            if args != None: fu = func(u, *args)
            else: fu = func(u)
        elif (cx - u) * (u - ulim) > 0.0:
            if args != None: fu = func(u, *args)
            else: fu = func(u)
            if fu < fc:
                bx, cx, u = cx, u, u + GOLD * (u - cx)
                if args != None: dum = func(u, *args)
                else: dum = func(u)
                fb, fc, fu = fc, fu, dum
        elif (u - ulim) * (ulim - cx) >= 0.0:
            u = ulim
            if args != None: fu = func(u, *args)
            else: fu = func(u)
        else:
            u = cx + GOLD * (cx - bx)
            if args != None: fu = func(u, *args)
            else: fu = func(u)

        # Shift the points to maintain bracketing
        ax, bx, cx = bx, cx, u
        fa, fb, fc = fb, fc, fu
        
    return ax, bx, cx

def powell(p, xi, func, tol=0.01, args=None):
    """
    Powell's optimization method.

    Parameters
    ----------
    p : np.ndarray
        Initial point, a 1D numpy array representing the starting point for optimization.
        
    xi : np.ndarray
        Matrix of search directions, a 2D numpy array where each column represents a direction.
        
    func : callable
        The function to minimize. It should accept a numpy array as input and return a scalar.
        
    tol : float, optional
        Tolerance for stopping condition (default is 0.01).
        
    args : tuple, optional
        Additional arguments to pass to `func`. Defaults to None.

    Returns
    -------
    np.ndarray
        The point at which the function is minimized.
    
    Raises
    ------
    ValueError
        If the maximum number of iterations is exceeded.
    """
    # Initialize points
    pt = np.copy(p)
    ptt = np.zeros_like(p)
    xit = np.zeros_like(p)
    fret = func(p) if args is None else func(p, *args)

    # Iterative process
    for iter_count in range(ITMAX + 1):
        fp = fret
        ibig = 0
        del_max = 0.0

        for i in range(len(p)):
            # Copy the i-th direction into xit
            xit[:] = xi[:, i]
            fptt = fret

            # Perform line minimization along direction xit
            p, xit, fret = lin_min(p, xit, func, args)

            # Check if the function value changes significantly
            if np.abs(fptt - fret) > del_max:
                del_max = np.abs(fptt - fret)
                ibig = i

        # Stopping condition
        if 2.0 * np.abs(fp - fret) <= tol * (np.abs(fp) + np.abs(fret)):
            return p

        # Check if maximum iterations exceeded
        if iter_count == ITMAX:
            raise ValueError("Powell exceeding maximum iterations.")

        # Extrapolate to a new point
        for j in range(len(p)):
            ptt[j] = 2.0 * p[j] - pt[j]
            xit[j] = p[j] - pt[j]
            pt[j] = p[j]

        fptt = func(ptt) if args is None else func(ptt, *args)

        # If the new point is better, update the directions
        if fptt < fp:
            t = 2.0 * (fp - 2.0 * fret + fptt) * (fp - fret - del_max)**2 - del_max * (fp - fptt)**2
            if t < 0.0:
                p, xit, fret = lin_min(p, xit, func, args)
                xi[:, ibig] = xi[:, -1]
                xi[:, -1] = xit

    return p

def lin_min(p, xi, func, args):
    """
    Line minimization along the direction `xi`, updating point `p`.

    Parameters
    ----------
    p : np.ndarray
        Initial point, a 1D numpy array representing the starting point for optimization.
        
    xi : np.ndarray
        Direction vector, a 1D numpy array indicating the search direction for minimization.
        
    func : callable
        The objective function to minimize. It should accept a numpy array as input and return a scalar.
        
    args : tuple, optional
        Additional arguments to pass to `func`. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing:
        - Updated point `p` after minimization.
        - Updated direction vector `xi` scaled by the step size.
        - The minimized function value at the new point.
    """
    # Step 1: Set global variables for use in f1dim
    pcom = np.copy(p)
    xicom = np.copy(xi)
    nrfunc = func

    # Step 2: Minimize along the direction xi using Brent's method
    ax = 0.0
    xx = 1.0
    xmin = brent(ax, xx, f1dim, args=(pcom, xicom, nrfunc, args))
    fret = f1dim(xmin, pcom, xicom, nrfunc, args)

    # Step 3: Update p and xi with the minimized step size (xmin)
    for j in range(len(p)):
        xi[j] *= xmin
        p[j] += xi[j]

    return p, xi, fret

def f1dim(x, pcom, xicom, nrfunc, args):
    """
    One-dimensional function for line minimization.

    This function evaluates the objective function at a point defined by moving
    along a direction vector from an initial point, scaled by a scalar input.

    Parameters
    ----------
    x : float
        Scalar input representing the step size along the direction defined by `xicom`.
        
    pcom : np.ndarray
        Initial point in the parameter space as a 1D numpy array.
        
    xicom : np.ndarray
        Direction vector as a 1D numpy array indicating the search direction for minimization.
        
    nrfunc : callable
        The objective function to minimize. It should accept a numpy array as input and return a scalar.
        
    args : tuple, optional
        Additional arguments to pass to `nrfunc`. Defaults to None.

    Returns
    -------
    float
        The function value at the new point calculated as `pcom + x * xicom`.
    """
    # Step 1: Calculate the new vector `xt` along the direction `xicom`
    xt = pcom + x * xicom  # Element-wise addition in numpy

    # Step 2: Evaluate the function at this new point
    f = nrfunc(xt) if args is None else nrfunc(xt, *args)

    # Step 3: Return the function value
    return f
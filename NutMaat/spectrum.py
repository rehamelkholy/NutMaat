import sys
import os
import numpy as np

from dataclasses import dataclass

@dataclass
class Results:
    """
    A class to store the results of the fitting procedure.

    Attributes
    ----------
    format_out : str
        The formatted string containing all the classification keywords of the spectrum.
        
    chi2 : float
        The chi-squared statistic representing the goodness of fit of the library match to the spectrum.
    """
    format_out: str
    chi2: float

class Spectrum:
    """
    A class to represent astronomical spectra and keep track of their attributes.

    This class initializes a spectrum object either from provided wavelength and flux arrays 
    or by reading from a specified file. It contains attributes for spectral properties, 
    quality flags, and additional parameters necessary for analysis.

    Parameters
    ----------
    wave : np.ndarray, optional
        A NumPy array containing the wavelength data of the spectrum. Default is `None`.
        
    flux : np.ndarray, optional
        A NumPy array containing the flux data corresponding to the wavelengths. Default is `None`.
        
    name : str, optional
        The name of the spectrum file (without file extension) or a descriptor for the spectrum. Default is an empty string.
        
    read_from_file : bool, optional
        A flag indicating whether to read the spectrum from a file. Default is `False`.
        
    path : str, optional
        The directory path where the spectrum file is located. Required if `read_from_file` is `True`.

    """
    def __init__(self, wave=None, flux=None, name="", read_from_file=False, path=None) -> None:
        if not read_from_file:
            self.name = name
            self.wave = self.wave_rebin = self.wave_out = wave
            self.flux = self.flux_rebin = self.flux_out = flux
        else:
            self.name = name[:name.find('.')]
            if path != None:
                self.path = os.path.join(path, name)
                self.name = os.path.splitext(os.path.basename(self.path))[0]
                in_file = self.path
            else:
                in_file = name
                self.name = os.path.splitext(os.path.basename(name))[0]
                
            self.wave, self.flux = get_spectrum(in_file)
            self.wave_rebin = self.wave_out = self.wave
            self.flux_rebin = self.flux_out = self.flux
            
        self.pcode = [16., 4.5]
        self.irbn = "t160l50p00.rbn"
        self.ispt = self.SPT = self.hSPT = self.mSPT = self.kSPT = self.LUM = self.PEC = ""
        self.spt = self.isp = self.sp_code = 0.
        self.lum = self.ilt = self.lum_code = 0.
        self.h_type = self.metal_type = 0.
        self.note = self.qual = ""
        self.done = self.flag_extra = self.Ba = self.pec = False
        self.Sr = self.Eu = self.Si = self.Cr = 0
        self.I = self.sf = self.flag_LB = self.iterate = 0
        self.NI = 1
        
        self.Iter = [Results(format_out="", chi2=np.inf) for _ in range(10)]
        self.cor_name = self.lib_match = None

def get_spectrum(in_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads in spectral data from an ASCII file while ignoring lines that start with '#'.
    
    Parameters
    ----------
    in_file : str
        The path to the spectrum file containing wavelength and flux data in two columns.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
            - wave: Array of wavelength values.
            - flux: Array of flux values.
    
    Raises
    ------
    FileNotFoundError
        If the specified input file cannot be found.
        
    ValueError
        If the data in the file cannot be converted to float or if the expected number of columns is not found.
    """
    wave, flux = [], []
    try:
        with open(in_file, "r") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                wave.append(float(parts[0]))
                flux.append(float(parts[1]))
    except FileNotFoundError:
        print("\nCannot find input spectrum file\n")
        sys.exit(1)
    
    return np.array(wave), np.array(flux)
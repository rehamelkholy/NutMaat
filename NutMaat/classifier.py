import logging
import os

from .library import *
from .spectrum import *
from .encoding import *
from .preprocessing import *
from .evaluate import *
from .peculiarities import *
from .utils import *
from ._version import __version__

class Classifier:
    """
    A class for classifying stellar spectra using a predefined spectral library.

    The `Classifier` class provides methods to classify stellar spectra by determining their spectral type and luminosity class. It sets up the necessary environment, including logging and file output configurations, and uses template-based and least squares methods for spectral classification.

    Parameters
    ----------
    lib : str, optional, default='libnor36'
        The name of the spectral library to be used for classification. This is a required
        input for the `Library` object that stores spectral templates and associated data.

    MKLIB : str, optional, default=''
        The path to the directory containing the spectral library files. If not provided,
        a default path within the `resources` directory is used.

    LOG : str, optional, default=None
        The path and filename for the log file. If not provided, a default 'LOG.log' file
        is created in the current working directory.

    out_file : str, optional, default=None
        The base name for the output file where classification results will be stored. If not provided, no output file is created. The output file will have a '.out' extension and be placed in the current working directory.
    """
    def __init__(self, lib: str = "libnor36", MKLIB : str = "", LOG : str = None, out_file : str = None) -> None:
        if not LOG: LOG = os.path.join(os.getcwd(), 'LOG')
        self.LOG = LOG + '.log'
        logging.basicConfig(filename=self.LOG, filemode='w', level=logging.INFO, format='%(message)s')
        
        if not out_file: self.out_file = None
        else:
            self.out_file = os.path.join(os.getcwd(), out_file) + '.out'
            open(self.out_file, 'w').close()
        
        print(f"NutMaat v{__version__}: {lib}")
        logging.info(f"NutMaat v{__version__}: {lib}")
        
        # Get variable MKLIB
        if MKLIB == "":
            MKLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
            logging.info(f"MKLIB not set: using default MKLIB path: {MKLIB}")
        else:
            logging.info(f"MKLIB read from environment as {MKLIB}")
        
        logging.info("================================================")
        
        self.library = Library(lib, MKLIB)
    
    def srebin(self, spectrum: Spectrum, spacing: float, output_file: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebins the wavelength and flux data of a given spectrum according to a specified spacing.

        This method modifies the `spectrum` object's wavelength and flux attributes by applying 
        a rebinning process. It can also write the rebinned data to an output file if specified.

        Parameters
        ----------
        spectrum : Spectrum
            An instance of the `Spectrum` class containing wavelength and flux data to be rebinned.
            
        spacing : float
            The new spacing between rebinned wavelength points.
            
        output_file : bool, optional
            If set to True, the method writes the rebinned wavelength and flux data to an output file. 
            The default is False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays:
            - The rebinned wavelength data.
            - The rebinned flux data.

        Raises
        ------
        Exception
            Raises an exception if there is an issue with file writing or rebinning.
        """
        spectrum.wave, spectrum.flux, _, _ = rebin(spectrum.wave, spectrum.flux, self.library.w_low, self.library.w_high, spacing)
        
        if output_file:
            if spectrum.name not in [None, ""]:
                rbn_file = os.path.join(os.getcwd(), spectrum.name + '.rbn')
            else:
                rbn_file = os.path.join(os.getcwd(), 'temp' + '.rbn')
            
            # Write rebinned data to output file
            with open(rbn_file, 'w') as out:
                for i in range(len(spectrum.wave)):
                    out.write(f"{spectrum.wave[i]:.6f} {spectrum.flux[i]:.6g}\n")
        
        return spectrum.wave, spectrum.flux
    
    def srebin_spectrum(self, spacing: float, df_output: bool = True, output_file: bool = False, from_df=False, df: pd.Series = None, cols: list[str] = None, file_name: str = None, from_file=False) -> pd.DataFrame:
        """
        Rebins the spectrum data from a DataFrame or a file, and optionally outputs the rebinned data as a DataFrame.

        This method allows for rebinned spectrum data to be created from either a pandas DataFrame 
        or a specified file. It can also write the rebinned data to a file and/or return the data 
        as a DataFrame based on the provided parameters.

        Parameters
        ----------
        spacing : float
            The new spacing between rebinned wavelength points.
            
        df_output : bool, optional
            If set to True, the method returns the rebinned data as a DataFrame. The default is True.
            
        output_file : bool, optional
            If set to True, the method writes the rebinned wavelength and flux data to an output file. 
            The default is False.
            
        from_df : bool, optional
            If True, indicates that the spectrum data is being sourced from a pandas DataFrame.
            
        df : pd.Series, optional
            The DataFrame containing the spectrum data, required if 'from_df' is True.
            
        cols : list[str], optional
            A list of column names to extract the wavelength and flux data from the DataFrame. 
            Required if 'from_df' is True.
            
        file_name : str, optional
            The name of the file to read the spectrum data from, required if 'from_file' is True.
            
        from_file : bool, optional
            If True, indicates that the spectrum data is being sourced from a file.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the rebinned wavelength and flux data if 'df_output' is True; 
            otherwise, returns None.

        Raises
        ------
        ValueError
            Raises an exception if the input conditions for using either DataFrame or file are not met.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_name is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        if from_df:
            if len(cols) > 2:
                spectrum = Spectrum(df[cols[1]], df[cols[2]], name=df[cols[0]])
            else:
                spectrum = Spectrum(df[cols[0]], df[cols[1]])
        else:
            spectrum = Spectrum(name=file_name, read_from_file=True)
        wave, flux = self.srebin(spectrum, spacing, output_file)
        
        if df_output:
            df_ = pd.DataFrame([{
                'name': spectrum.name,
                'wave': wave,
                'flux': flux
            }])
            
            return df_
        return None
    
    def srebin_spectra(self, spacing: float, df_output: bool = True, output_files: bool = False, from_df=False, df: pd.DataFrame = None, cols: list[str] = None, file_names: list[str] = None, from_file=False) -> pd.DataFrame:
        """
        Rebins multiple spectra from a DataFrame or a list of files, and optionally outputs the rebinned data as a DataFrame.

        This method processes and rebins spectra either from a provided pandas DataFrame or from a list of files. 
        It utilizes progress bars to indicate processing status and can write rebinned data to files and/or return the results as a DataFrame based on the specified parameters.

        Parameters
        ----------
        spacing : float
            The new spacing between rebinned wavelength points for all spectra.
            
        df_output : bool, optional
            If set to True, the method returns the rebinned data as a concatenated DataFrame. The default is True.
            
        output_files : bool, optional
            If set to True, the method writes each rebinned spectrum's wavelength and flux data to separate output files. 
            The default is False.
            
        from_df : bool, optional
            If True, indicates that the spectrum data is being sourced from a pandas DataFrame.
            
        df : pd.DataFrame, optional
            The DataFrame containing the spectrum data, required if 'from_df' is True.
            
        cols : list[str], optional
            A list of column names to extract the wavelength and flux data from the DataFrame. 
            Required if 'from_df' is True.
            
        file_names : list[str], optional
            A list of file names to read the spectrum data from, required if 'from_file' is True.
            
        from_file : bool, optional
            If True, indicates that the spectrum data is being sourced from files.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the concatenated rebinned wavelength and flux data if 'df_output' is True; 
            otherwise, returns None.

        Raises
        ------
        ValueError
            Raises an exception if the input conditions for using either DataFrame or file are not met.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_names is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        
        from tqdm import tqdm
        
        results = []
        
        if from_df:
            for _, row in tqdm(df.iterrows(), desc="Rebinning spectra", total=len(df)):
                result = self.srebin_spectrum(spacing, df_output, output_files, from_df, row, cols)
                results.append(result)
        else:
            for name in tqdm(file_names, desc="Rebinning spectra", total=len(file_names)):
                result = self.srebin_spectrum(spacing, df_output, output_files, from_df, df, cols, file_name=name, from_file=from_file)
                results.append(result)
        
        if df_output:
            df_results = pd.concat(results, ignore_index=True)
        
            return df_results
        return None
    
    def smooth(self, spectrum: Spectrum, input_spacing: float, output_spacing: float, resolution: float, output_file: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooths the flux values of a given spectrum using a Gaussian-like kernel 
        and outputs the smoothed spectrum at specified wavelengths.

        This method takes a `Spectrum` object and applies a smoothing algorithm to its flux values, 
        adjusting the wavelength spacing based on the specified input and output parameters. 
        The smoothing is performed using a kernel defined by the resolution and input spacing, 
        allowing for enhanced analysis of spectral data.

        Parameters
        ----------
        spectrum : Spectrum
            The `Spectrum` object containing wavelength and flux data to be smoothed.
            
        input_spacing : float
            The spacing between input wavelengths (in the same units as the wavelengths in `spectrum`).
            
        output_spacing : float
            The desired spacing between output wavelengths after smoothing.
            
        resolution : float
            The resolution of the smoothing process, which influences the kernel width.
            
        output_file : bool, optional
            If set to True, the smoothed spectrum will be written to a file. 
            Defaults to False, meaning no file output.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays: 
            - The first array is the new wavelength data after smoothing.
            - The second array is the corresponding smoothed flux values.
        """
        # Calculate necessary parameters
        npt = len(spectrum.wave) - 1  # Number of points
        resolution /= 2.0
        n = round(resolution / input_spacing)
        nspace = round(output_spacing / input_spacing)
        a = -math.log10(0.50) / (n ** 2)
        
        # Initialize output wavelength
        wave = spectrum.wave[0]
        
        new_wave, new_flux = [], []
        
        i = 0
        while i <= npt:
            low = max(0, i - 3 * n)
            high = min(npt, i + 3 * n)
            sum1, sum2 = 0.0, 0.0
            for k in range(low, high + 1):
                z = pow(10.0, - a * (k - i) ** 2)
                sum1 += spectrum.flux[k] * z
                sum2 += z
            
            Ds = sum1 / sum2
            new_wave.append(wave)
            new_flux.append(Ds)
            wave += output_spacing
            i += nspace
        
        spectrum.wave, spectrum.flux = np.array(new_wave), np.array(new_flux)
        
        if output_file:
            # Open output file for writing the smoothed spectrum
            if spectrum.name not in [None, ""]:
                out_file = os.path.join(os.getcwd(), spectrum.name+'.smz')
            else:
                out_file = os.path.join(os.getcwd(),"temp.smz")
            
            with open(out_file, 'w') as fp:
                for w, f in new_wave, new_flux:
                    fp.write(f"{w:8.3f}  {f:g}\n")
                    
        return spectrum.wave, spectrum.flux
    
    def smooth_spectrum(self, input_spacing: float, output_spacing: float, resolution: float, df_output: bool = True, output_file: bool = False, from_df=False, df: pd.Series = None, cols: list[str] = None, file_name: str = None, from_file=False) -> pd.DataFrame:
        """
        Smooths the flux values of a spectrum and outputs the results either 
        as a DataFrame or to a file, based on the specified input parameters.

        This method can generate a `Spectrum` object from either a DataFrame 
        or a file, applies smoothing to the flux values using a Gaussian-like kernel, 
        and optionally saves the smoothed data. The method ensures that input data 
        is correctly processed and provides flexibility in data sources.

        Parameters
        ----------
        input_spacing : float
            The spacing between input wavelengths (in the same units as the wavelengths in the spectrum).
            
        output_spacing : float
            The desired spacing between output wavelengths after smoothing.
            
        resolution : float
            The resolution for the smoothing process, which affects the kernel width.
            
        df_output : bool, optional
            If True, the smoothed spectrum will be returned as a DataFrame. Defaults to True.
            
        output_file : bool, optional
            If True, the smoothed spectrum will be written to a file. Defaults to False.
            
        from_df : bool, optional
            Indicates the source of the spectrum data; if True, the spectrum is created from a DataFrame.
            Defaults to False.
            
        df : pd.Series, optional
            The DataFrame series containing the spectrum data, required if `from_df` is True.
            
        cols : list[str], optional
            List of column names from the DataFrame to use for wavelength and flux data,
            required if `from_df` is True.
            
        file_name : str, optional
            The name of the file from which to read the spectrum, required if `from_file` is True.
            
        from_file : bool, optional
            Indicates whether to read the spectrum data from a file. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the smoothed wavelength and flux data, 
            or None if df_output is set to False.

        Raises
        ------
        ValueError
            If the conditions for `from_df` and `from_file` are not met,
            or if necessary parameters are missing.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_name is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        
        if from_df:
            if len(cols) > 2:
                spectrum = Spectrum(df[cols[1]], df[cols[2]], name=df[cols[0]])
            else:
                spectrum = Spectrum(df[cols[0]], df[cols[1]])
        else:
            spectrum = Spectrum(name=file_name, read_from_file=True)
        
        wave, flux = self.smooth(spectrum, input_spacing, output_spacing, resolution, output_file)
        
        if df_output:
            df_ = pd.DataFrame([{
                'name': spectrum.name,
                'wave': wave,
                'flux': flux
            }])
            
            return df_
        return None
    
    def smooth_spectra(self, input_spacing: float, output_spacing: float, resolution: float, df_output: bool = True, output_files: bool = False, from_df=False, df: pd.DataFrame = None, cols: list[str] = None, file_names: list[str] = None, from_file=False) -> pd.DataFrame:
        """
        Smooths multiple spectra and outputs the results either as a DataFrame or 
        to individual files, based on the specified input parameters.

        This method processes multiple spectra, applying smoothing to the flux values 
        using a Gaussian-like kernel. The spectra can be provided either from a DataFrame 
        or as file names. The function utilizes progress tracking to inform the user 
        of the smoothing process's progress.

        Parameters
        ----------
        input_spacing : float
            The spacing between input wavelengths (in the same units as the wavelengths in the spectrum).
            
        output_spacing : float
            The desired spacing between output wavelengths after smoothing.
            
        resolution : float
            The resolution for the smoothing process, affecting the kernel width.
            
        df_output : bool, optional
            If True, the smoothed spectra will be returned as a DataFrame. Defaults to True.
            
        output_files : bool, optional
            If True, the smoothed spectra will be written to individual output files. Defaults to False.
            
        from_df : bool, optional
            Indicates the source of the spectrum data; if True, the spectra are created from a DataFrame.
            Defaults to False.
            
        df : pd.DataFrame, optional
            The DataFrame containing the spectrum data, required if `from_df` is True.
            
        cols : list[str], optional
            List of column names from the DataFrame to use for wavelength and flux data,
            required if `from_df` is True.
            
        file_names : list[str], optional
            A list of file names from which to read the spectrum data, required if `from_file` is True.
            
        from_file : bool, optional
            Indicates whether to read the spectrum data from files. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the smoothed wavelengths and flux data, 
            or None if df_output is set to False.

        Raises
        ------
        ValueError
            If the conditions for `from_df` and `from_file` are not met,
            or if necessary parameters are missing.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_names is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        
        from tqdm import tqdm
        
        results = []
        
        if from_df:
            for _, row in tqdm(df.iterrows(), desc="Smoothing spectra", total=len(df)):
                result = self.smooth_spectrum(input_spacing, output_spacing, resolution, df_output, output_files, from_df, row, cols)
                results.append(result)
        else:
            for name in tqdm(file_names, desc="Smoothing spectra", total=len(file_names)):
                result = self.smooth_spectrum(input_spacing, output_spacing, resolution, df_output, output_files, from_df, df, cols, file_name=name, from_file=from_file)
                results.append(result)
        
        if df_output:
            df_results = pd.concat(results, ignore_index=True)
        
            return df_results
        return None
     
    def classify_spectrum(self, rough_flag: int, NI: int = 1, df_output: bool = True, output_files: bool = False, from_df=False, df: pd.Series = None, cols: list[str] = None, file_name: str = None, from_file=False) -> pd.DataFrame:
        """
        Classifies a spectrum based on provided parameters and outputs the results 
        either as a DataFrame or to individual files, depending on the input method.

        This method allows users to classify spectra either from a DataFrame or 
        from file input, using a classification algorithm defined in the 
        `classify` method. The classification can be performed with a rough 
        flag indicating the classification type, and the number of iterations 
        (NI) can be adjusted.

        Parameters
        ----------
        rough_flag : int
            An integer indicating the classification method to be used.
            This may define a rough classification approach.
            
        NI : int, optional
            The number of iterations for the classification process.
            Defaults to 1.
            
        df_output : bool, optional
            If True, the classification results will be returned as a DataFrame. Defaults to True.
            
        output_files : bool, optional
            If True, the classification results will be written to individual output files. Defaults to False.
            
        from_df : bool, optional
            Indicates whether to create the spectrum from a DataFrame. Defaults to False.
            
        df : pd.Series, optional
            A pandas Series containing the spectrum data, required if `from_df` is True.
            
        cols : list[str], optional
            A list of column names from the DataFrame to use for wavelength and flux data,
            required if `from_df` is True.
            
        file_name : str, optional
            The name of the file from which to read the spectrum data, required if `from_file` is True.
            
        from_file : bool, optional
            Indicates whether to read the spectrum data from a file. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the classification results,
            or None if df_output is set to False.

        Raises
        ------
        ValueError
            If the conditions for `from_df` and `from_file` are not met,
            or if necessary parameters are missing.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_name is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        
        if from_df:
            if len(cols) > 2:
                spectrum = Spectrum(df[cols[1]], df[cols[2]], name=df[cols[0]])
            else:
                spectrum = Spectrum(df[cols[0]], df[cols[1]])
        else:
            spectrum = Spectrum(name=file_name, read_from_file=True)
        
        return self.classify(spectrum, rough_flag, NI, df_output, output_files)

    def classify_spectra(self, rough_flag: int, NI: int = 1, df_output: bool = True, output_files: bool = False, from_df=False, df: pd.DataFrame = None, cols: list[str] = None, file_names: list[str] = None, from_file=False) -> pd.DataFrame:
        """
        Classifies multiple spectra based on provided parameters and outputs the 
        results either as a DataFrame or to individual files, depending on the input method.

        This method allows users to classify multiple spectra either from a DataFrame
        or from file inputs, using a classification algorithm defined in the 
        `classify` method. The classification can be performed with a rough flag 
        indicating the classification type, and the number of iterations (NI) 
        can be adjusted.

        Parameters
        ----------
        rough_flag : int
            An integer indicating the classification method to be used.
            This may define a rough classification approach.
            
        NI : int, optional
            The number of iterations for the classification process.
            Defaults to 1.
            
        df_output : bool, optional
            If True, the classification results will be returned as a DataFrame. Defaults to True.
            
        output_files : bool, optional
            If True, the classification results will be written to individual output files. Defaults to False.
            
        from_df : bool, optional
            Indicates whether to create the spectra from a DataFrame. Defaults to False.
            
        df : pd.DataFrame, optional
            A pandas DataFrame containing the spectrum data, required if `from_df` is True.
            
        cols : list[str], optional
            A list of column names from the DataFrame to use for wavelength and flux data,
            required if `from_df` is True.
            
        file_names : list[str], optional
            A list of file names from which to read the spectrum data, required if `from_file` is True.
            
        from_file : bool, optional
            Indicates whether to read the spectrum data from files. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the classification results,
            or None if df_output is set to False.

        Raises
        ------
        ValueError
            If the conditions for `from_df` and `from_file` are not met,
            or if necessary parameters are missing.
        """
        if not ((from_df or from_file) and (not (from_df and from_file))):
            raise ValueError("One and only one of 'from_df' or 'from_file' must be True.")
        elif from_df and (df is None or cols is None):
            raise ValueError("Both 'df' and 'cols' must be provided when 'from_df' is True.")
        elif from_file and file_names is None:
            raise ValueError("'file_name' must be provided when 'from_file' is True.")
        
        from tqdm import tqdm
        
        results = []
        
        if from_df:
            if len(cols) > 2:
                for _, row in tqdm(df.iterrows(), desc="Classifying spectra", total=len(df)):
                    spectrum = Spectrum(row[cols[1]], row[cols[2]], name=row[cols[0]])
                    result = self.classify(spectrum, rough_flag, NI, df_output, output_files)
                    results.append(result)
            else:
                for _, row in tqdm(df.iterrows(), desc="Classifying spectra", total=len(df)):
                    spectrum = Spectrum(row[cols[0]], row[cols[1]])
                    result = self.classify(spectrum, rough_flag, NI, df_output, output_files)
                    results.append(result)
        else:
            for name in tqdm(file_names, desc="Classifying spectra", total=len(file_names)):
                spectrum = Spectrum(name=name, read_from_file=True)
                result = self.classify(spectrum, rough_flag, NI, df_output, output_files)
                results.append(result)
        
        if df_output:
            df_results = pd.concat(results, ignore_index=True)
            
            return df_results
        return None
    
    def classify(self, spectrum: Spectrum, rough_flag: int, NI=1, df_output=True, output_files=False) -> pd.DataFrame:
        """
        Classifies the provided spectrum by determining its spectral type and luminosity class.

        This method performs a detailed classification of a given stellar spectrum. 
        It first determines a rough spectral type using initial methods, followed by 
        a series of refinements to optimize the spectral type and luminosity class 
        using a template-based and least squares approach. The classification can 
        be iterated multiple times to improve classification quality.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object to classify, containing relevant data for classification.
        
        rough_flag : int
            Indicates the method for obtaining the initial rough spectral type. 
            - `1`: Uses `rough_type_1` for rectified spectra.
            - `2`: Uses `rough_type_2` for flux-calibrated spectra.

        NI : int, optional, default=1
            Number of iterations for refining the classification. If set to more than 
            1, the method will attempt additional iterations to enhance the 
            classification accuracy.
        
        df_output : bool, optional, default=True
            Determines if the output should be formatted as a DataFrame. If `True`, 
            the function returns a DataFrame with classification results.

        output_files : bool, optional, default=False
            If `True`, the method generates output files for the corrected spectrum 
            (with a `.cor` extension) and the library match (with a `.mat` extension). 
            The filenames are derived from the spectrum name.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the classification results if `df_output` is set to 
            `True`, or `None` if `df_output` is `False`.

        Raises
        ------
        ValueError
            If the input spectrum does not conform to the library specifications, or 
            if the standards library cannot be found.
        """
        logging.info(f"Classifying {spectrum.name}\nRough type = {rough_flag}")
        
        if output_files:
            # Sets the names for the various output files.
            file_name = (spectrum.name if spectrum.name != "" else "spectrum")
            spectrum.cor_name = file_name + '.cor'
            spectrum.lib_match = file_name + '.mat'
        else: spectrum.cor_name = spectrum.lib_match = None
        
        if NI > 1:
            spectrum.NI = NI
            flagi = True
        else:
            flagi = False
            
        if flagi:
            spectrum.isp = spt_code(spectrum.ispt)
            spectrum.pcode[0] = spectrum.isp
            spectrum.irbn = "t{:02.0f}0l50p00.rbn".format(spectrum.isp)
        
        # Check to see if input spectrum conforms to library specifications #
        n_err = lib_conform(self.library, spectrum)
        if n_err in [1,2]:
            logging.info("Wavelength range does not conform to library specifications")
            self.write_to_out(f'{spectrum.name} Cannot classify -- see log\n')
            if not self.out_file:
                print(f'{spectrum.name} Cannot classify -- see log\n')
            if df_output: return format_df(spectrum)
            return None
        
        if not self.library.flag_lib:
            print("\nCannot find standards library\n")
            if df_output: return format_df(spectrum)
            return None
        
        logging.info(f"prelim = {self.library.prelim}")
        
        '''
        Now get a rough initial type.  If you are classifying a rectified spectrum, use rough_type_1, if a flux-calibrated spectrum, rough_type_2 is better, but experiment to see which works for your spectra
        '''
        if rough_flag == 1:
            rough_type_1(self.library, spectrum)
        elif rough_flag == 2:
            rough_type_2(self.library, spectrum)
        
        # Format the file name
        spectrum.irbn = f"t{10 * spectrum.isp:03.0f}l{spectrum.ilt:1.0f}0p00.rbn"
        
        logging.info(f"Initial type = {spectrum.irbn}")
        
        spectrum.wave_rebin, spectrum.flux_rebin, ind_start, ind_end = rebin(spectrum.wave, spectrum.flux, self.library.w_low, self.library.w_high, self.library.space)
        
        '''
        After the rough type, we really need to check to see if star is normal. We first do a preliminary check to see if the star is an emission-line star. Then we pass the star to the normality-checking routine DetectNN2.  At the moment, we are doing this only if rough_flag is 2.
        '''
        e = emission(spectrum.wave_rebin, spectrum.flux_rebin)
        if spectrum.isp >= 34 and e == 0:
            # Check to see if star is carbon star
            ratio = Carbon(spectrum.wave_rebin, spectrum.flux_rebin)
            logging.info(f"Carbon ratio = {ratio:.6f}")
            if ratio >= 3.:
                logging.info("Classified as a Carbon Star")
                self.write_to_out(f"{spectrum.name}   | Carbon star\t\t|         | \\\\ \n")
                logging.info("================================================")
                if df_output: return format_df(spectrum)
                return None
            else:
                normal = 0
                spectrum.note = " \\\\ "
        else:
            normal = detect_NN(self.library, spectrum)
            spectrum.note = " \\\\ "
            if normal == 14:
                self.write_to_out(f"{spectrum.name}  | Unclassifiable             |         | \\\\ \n")
                if df_output: return format_df(spectrum)
                return None
            if normal == 13: spectrum.note = " ? \\\\ "
            if normal not in [0, 7, 13]:
                logging.info(f"Star is not normal: code = {normal}")
                classifications = {
                    1: "DA",
                    2: "DB",
                    3: "DO",
                    4: "DZ",
                    5: "DQ",
                    8: "Carbon star",
                    9: "emission-line?",
                    10: "WN",
                    11: "Helium nova",
                    12: "WC",
                    6: "??"
                }
                if normal in classifications:
                    self.write_to_out(f"{spectrum.name}  | {classifications[normal]}\t\t|         | \\\\\n")
                logging.info("================================================\n")
                if df_output: return format_df(spectrum)
                return None
            else: logging.info("Star appears normal")

        # Pass the spectrum through mk_prelim #
        mk_prelim(self.library, spectrum)
        spectrum.spt = spectrum.isp
        spectrum.lum = spectrum.ilt
        '''
        Now correct energy distribution to initial type, provided the library specifies a template should be used.  Rectified libraries should not apply templates.
        '''
        if spectrum.spt <= self.library.s_cool and self.library.flag_template:
            logging.info("Spectral template applied")
            wave_temp, flux_temp = sp2class(self.library, spectrum)
            spectrum.flux_out = template_DSO(spectrum.wave_out, spectrum.flux_out, wave_temp, flux_temp)
        
        # Now refine initial type using least squares with Powell method #
        spectrum.pcode = [spectrum.isp, spectrum.ilt]
        spectrum.pcode = powell(spectrum.pcode, np.eye(2), spt2min, args=(self.library, spectrum))
        
        spectrum.spt = spectrum.sp_code = spectrum.pcode[0]
        spectrum.lum = spectrum.lum_code = spectrum.pcode[1]

        if spectrum.spt >= self.library.s_cool:
            logging.info(f"Initial spectral type {spectrum.spt:.6f} is later than library limit.  Adjusting to within library range")
            spectrum.spt = spectrum.sp_code = self.library.s_cool - 3.
            logging.info(f"Initial spectral type adjusted to {spectrum.spt:.6f}")
        
        if abs(spectrum.lum) > 5.:
            spectrum.lum = spectrum.lum_code = 5.

        spectrum.SPT = code2spt(spectrum.spt)
        spectrum.LUM = code2lum(spectrum.lum)
        logging.info(f"spt = {spectrum.spt:.6f} lum = {spectrum.lum:.6f}")
        
        self.write_to_out(f"{spectrum.name}  | ")
        # flag_name = 1?
        logging.info(f"Initial Spectral type estimate = {spectrum.SPT} {spectrum.LUM}")
        
        spectrum.I += 1
        while spectrum.I <= spectrum.NI:
            spectrum.iterate = 0
            # If quality is poor or fair, add one iteration #
            if spectrum.I == spectrum.NI and ("fair" in spectrum.qual or "poor" in spectrum.qual) and not spectrum.flag_extra:
                logging.info("Going for an extra iteration because of low quality")
                spectrum.NI += 1
                spectrum.flag_extra = True
            
            # If iterating, use the matched spectrum as a template to correct the energy distribution #
            spectrum.lum = spectrum.lum_code
            spectrum.spt = spectrum.sp_code
            
            if abs(spectrum.lum) > 5.2: spectrum.lum = 5.2
            if spectrum.I >= 2 and spectrum.spt <= self.library.s_cool and self.library.flag_template:
                logging.info(f"Spectral template spt = {spectrum.spt:.1f}  lum = {spectrum.lum:.1f}\n")
                wave_temp, flux_temp = sp2class(self.library, spectrum)
                spectrum.flux_out = template_DSO(spectrum.wave_out, spectrum.flux_out, wave_temp, flux_temp)
            
            # Print out the flux-corrected spectrum into name.cor if spt <= 39, or the uncorrected if spt > 39 #
            if spectrum.cor_name and spectrum.I >= 2 and self.library.flag_template:
                with open(spectrum.cor_name, "w") as cor:
                    for i in range(len(spectrum.wave_out)):
                        cor.write(f"{spectrum.wave_out[i]} {spectrum.flux_out[i]}\n")
            
            if spectrum.spt < 5. and not spectrum.done: self.class_O(spectrum)
            elif 5. <= spectrum.spt < 15. and not spectrum.done: self.class_B(spectrum)
            elif 15. <= spectrum.spt < 23 and not spectrum.done: self.class_A(spectrum)
            elif 23. <= spectrum.spt <= 34. and not spectrum.done: self.class_FG(spectrum)
            elif spectrum.spt > 34. and not spectrum.done: self.class_KM(spectrum)
            spectrum.I += 1
        
        logging.info("")
        for i in range(1, spectrum.NI+1):
            logging.info(f"{i}: {spectrum.Iter[i].format_out}   {spectrum.Iter[i].chi2:.1e}")
        
        logging.info("================================================\n")
        if df_output: return format_df(spectrum)
        return None
    
    def class_O(self, spectrum: Spectrum) -> None:
        """
        Classifies a given spectrum as an O-type star and updates the spectrum object accordingly.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object to be classified.
        """
        logging.info("Classifying as an O-type star")
        library = self.library
        
        spectrum.lum_code = spectrum.lum
        spt_heII = brent(spectrum.spt - 1, spectrum.spt, spt_HeII, args=(library, spectrum))        
        logging.info(f"HeII type = {spt_heII:.6f}")
        spectrum.sp_code = spectrum.spt = spt_heII
        
        if spectrum.sp_code >= 6.5:
            self.class_B(spectrum)
            return
        
        spectrum.SPT = code2spt(spectrum.spt)
        spectrum.LUM = code2lum(spectrum.lum)
        
        CHI2 = match(library, spectrum)
        spectrum.Iter[spectrum.I].chi2 = CHI2
        spectrum.qual = quality(spectrum, CHI2)
        if spectrum.I <= spectrum.NI:
            to_out = f"{spectrum.SPT} {spectrum.LUM}"
            to_out_length = len(to_out)

            if to_out_length <= 9:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t\t\t{spectrum.qual}"
            elif 9 < to_out_length <= 17:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t\t{spectrum.qual}"
            else:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t{spectrum.qual}"
        if spectrum.I == spectrum.NI:
            J = find_best(spectrum.Iter, spectrum.NI)
            self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
            logging.info(f"\nBest iteration: I = {J}")
            spectrum.done = True
        
        logging.info(f"{spectrum.I}: {spectrum.SPT} {spectrum.LUM}  {CHI2:7.1e}")
        return

    def class_B(self, spectrum: Spectrum) -> None:
        """
        Classifies a given spectrum as a B-type star and performs detailed analysis to refine the classification.
        
        Parameters
        ----------
        spectrum : Spectrum
            An instance of the Spectrum class containing the observed spectrum data to be classified.
        """
        logging.info("Classifying as a B-type star")
        
        library = self.library
        spectrum.iterate += 1
        
        # First, check to make certain that this is not really an A-type star by looking at the Ca K-line #
        spt_K = brent(spectrum.spt - 1, spectrum.spt, spt_CaK, args=(library, spectrum))
        logging.info(f"K-line type = {spt_K:.6f}")
        
        if spt_K >= 15.:
            spectrum.spt = spt_K
            self.class_A(spectrum)
            return
        
        # Ensure this is not a DA star with very wide Hydrogen lines #
        D2 = hyd_D2(spectrum.wave_out, spectrum.flux_out)
        logging.info(f"D2 = {D2:4.1f}")
        if D2 >= 30.:
            self.write_to_out(f" DA\t\t\t\t|         |\\\\\n")
            logging.info(f"D2 = {D2:4.1f} DA")
            spectrum.I = spectrum.NI
            return
        
        # Let us now check that this is not an O-type star by examining an He II line #
        spt_heII = brent(spectrum.spt - 1, spectrum.spt, spt_HeII, args=(library, spectrum))
        logging.info(f"HeII type = {spt_heII:.6f}")
        spectrum.sp_code = spectrum.spt = spt_heII
        if spt_heII <= 6.5:
            spectrum.spt = spt_heII
            self.class_O(spectrum)
            return
        
        # We now determine an approximate temperature type based on Helium I strengths and metallic-line ratios (more appropriate for early-B type stars #
        spectrum.spt = brent(spectrum.spt - 1, spectrum.spt, spt_HeImet, args=(library, spectrum))
        logging.info(f"Helium/metal spectral type = {spectrum.spt:.6f}")
        spectrum.sp_code = spectrum.spt
        
        # Now improve the luminosity type
        spectrum.lum = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
        if abs(spectrum.lum) > 5.2: spectrum.lum = 5.2
        
        logging.info(f"Luminosity type = {spectrum.lum:.6f}")
        
        # Polish the temperature type
        spectrum.lum_code = spectrum.lum
        
        spectrum.spt = brent(spectrum.spt - 1, spectrum.spt, spt_HeImet, args=(library, spectrum))        
        logging.info(f"Polished helium/metal spectral type = {spectrum.spt:.6f}")
        
        # Now check for helium peculiarities
        diff = HeI_pec(library, spectrum)
        
        logging.info(f"Helium strength = {diff:.6f}")
        
        if diff <= -0.1: spectrum.PEC = "Helium weak"
        if diff >= 0.1: spectrum.PEC = "Helium strong"
        
        # Check for any other obvious peculiarities
        spectrum.pec = peculiarity(library, spectrum)
        
        # Check to see if the star is a Lambda Boo star by looking at the Mg II 4481 line #
        if spectrum.spt >= 13.5 and not spectrum.pec:
            mg = MgII(library, spectrum)
            logging.info(f"mg = {mg:.6f}")
            if mg <= -0.005:
                logging.info("This looks like it could be a Lambda Boo star")
                spectrum.flag_LB = lam_boo(library, spectrum)
                if spectrum.flag_LB == 1:
                    logging.info("Yes, it is a Lambda Boo star")
                    spectrum.SPT = spectrum.hSPT = code2spt(spectrum.h_type)
                    spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                    logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo")
                    
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                    spectrum.Iter[spectrum.I].chi2 = 1.0
                    
                    if spectrum.I == spectrum.NI:
                        to_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                        to_out_length = len(to_out)
                        self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo  \t\t\t|          | {spectrum.note}\n", to_out_length)                      
                    return
                else:
                    logging.info("No, it is not a Lambda Boo")
        
        spectrum.pec = peculiarity(library, spectrum)
        
        if spectrum.pec:
            PEC = " "
            if spectrum.Sr == 1: PEC += "(Sr)"
            elif spectrum.Sr == 2: PEC += "Sr"
            if spectrum.Si == 1: PEC += "Si"
            if spectrum.Eu == 1: PEC += "Eu"
            if spectrum.Cr == 1: PEC += "Cr"
        
        spectrum.SPT = code2spt(spectrum.spt)
        spectrum.LUM = code2lum(spectrum.lum)
        
        CHI2 = match(library, spectrum)
        spectrum.qual = quality(spectrum, CHI2)
        spectrum.Iter[spectrum.I].chi2 = CHI2
        
        if spectrum.I <= spectrum.NI:
            to_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC}"
            to_out_length = len(to_out)

            if to_out_length <= 9:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t\t{spectrum.qual}"
            elif 9 < to_out_length <= 17:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t{spectrum.qual}"
            else:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t{spectrum.qual}"
        
        logging.info(f"{spectrum.I}: {spectrum.SPT} {spectrum.LUM} {spectrum.PEC}  {CHI2:7.1e}")
        if spectrum.I == spectrum.NI:
            # Repress hypergiants, as the code is not competent to classify them #
            if spectrum.lum < 0.0:
                self.write_to_out("Unclassifiable             |         | \\\\ \n")
            else:
                J = find_best(spectrum.Iter, spectrum.NI)
                self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                logging.info(f"\nBest iteration: I = {J}")
                spectrum.done = True
        return
                    
    def class_A(self, spectrum: Spectrum) -> None:
        """
        Classifies a star as an A-type star based on its spectrum data.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object containing the star's spectral data, including wavelength, flux, preliminary spectral type, luminosity, and other relevant attributes.
        """
        library = self.library
        
        spectrum.sf = 2
        logging.info("\nClassifying this star as an A-type star")
        spectrum.iterate += 1
        
        # Ensure it is not a DA star with very broad hydrogen lines #
        D2 = hyd_D2(spectrum.wave_out, spectrum.flux_out)
        logging.info(f"D2 = {D2:4.1f}")
        if D2 >= 30.:
            self.write_to_out("DA \t\t\t|          | \\\\ \n")
            logging.info(f"D2 = {D2:4.1f}  DA")
            spectrum.I = spectrum.NI
            return
        
        # We first begin by improving the luminosity type assuming the preliminary spectral type. This can help in the case of stars in which the hydrogen-line profiles have been poorly rectified #
        spectrum.lum = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
        
        if abs(spectrum.lum) > 5.2: spectrum.lum = 5.2
        logging.info(f"lum = {spectrum.lum:.6f}")
        
        # We also check at this point for obvious peculiarities #
        spectrum.pec = peculiarity(library, spectrum)
        
        # Early A-type stars #
        flag_vsini = 0
        flag_early = False
        if spectrum.spt <= 19.5:
            flag_early = True
            # Determine the K-line type
            spt_K = brent(spectrum.spt - 1, spectrum.spt, spt_CaK, args=(library, spectrum))
            logging.info(f"K-line type = {spt_K:.6f}")
            
            spt_m = brent(spt_K - 1, spt_K, spt_metal, args=(library, spectrum))
            logging.info(f"Metallic-line type = {spt_m:.6f}")
            
            # if the resolution is poor, the metallic-line type can be faulty, so if spt_m < spt_K, set spt_m = spt_K #
            if spt_m < spt_K - 2:
                spt_m = spt_K
                logging.info("Correcting metallic-line type to K-line type")
            
            # Check to see if the star is a metal-weak A-type star by looking at the Mg II 4481 line #
            mg = MgII(library, spectrum)
            logging.info(f"mg = {mg:.6f}")
            spectrum.flag_LB = 0
            if mg <= -0.005 and not spectrum.pec:
                logging.info("This looks like it could be a metal-weak star")
                spectrum.flag_LB = lam_boo_2(library, spectrum)
            if spectrum.flag_LB == 1:
                logging.info("It appears to be a Lambda Boo star")
                spectrum.SPT = spectrum.hSPT = code2spt(spectrum.h_type)
                spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                
                logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo")
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                spectrum.Iter[spectrum.I].chi2 = 1. # fake chi2 for peculiar star
                
                if spectrum.I == spectrum.NI:
                    to_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                    to_out_length = len(to_out)
                    self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo \t\t\t|         |\\\\\n", to_out_length)
                return
            elif mg <= -0.005 and spectrum.flag_LB == 0:
                logging.info("No, it is not metal-weak; it may be a luminous A-type star")
            
            if spectrum.flag_LB == 2:
                logging.info("It appears to be a metal-weak star")
                spectrum.SPT = spectrum.hSPT = code2spt(spectrum.h_type)
                spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                spectrum.LUM = code2lum(spectrum.lum_code)
                
                logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak")
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak"
                spectrum.Iter[spectrum.I].chi2 = 1. # fake chi2 for peculiar star
                
                if spectrum.I == spectrum.NI:
                    to_out = f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak"
                    to_out_length = len(to_out)
                    self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak\t\t\t|         | {spectrum.note}\n", to_out_length)
                return
            elif mg <= -0.005 and spectrum.flag_LB == 0:
                logging.info("No, it is not metal-weak")
            
            if spectrum.flag_LB == 3:
                logging.info("This star is probably rapidly rotating")
                flag_vsini = 1
        
            Am_flag = 0
            if spt_m - spt_K > 2.: Am_flag = 1
            # Because it is difficult to determine spectral type from the hydrogen lines alone if star is early A-type #
            spt_h = (spt_m + spt_K) / 2.
        # Late A-type star: Determine H-line type, assuming preliminary luminosity type #
        else:
            # Check to see if the star is a Lambda Boo star by looking at the Mg II 4481 line #
            mg = MgII(library, spectrum)
            logging.info(f"mg = {mg:.6f}")
            if mg <= -0.005 and not spectrum.pec:
                logging.info("This looks like it could be a metal-weak star")
                spectrum.flag_LB = lam_boo_2(library, spectrum)
                logging.info(f"flag_LB = {spectrum.flag_LB}")
            if spectrum.flag_LB == 1:
                logging.info("It appears to be a Lambda Boo star")
                spectrum.SPT = spectrum.hSPT = code2spt(spectrum.h_type)
                spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                
                logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo")
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                spectrum.Iter[spectrum.I].chi2 = 1. # fake chi2 for peculiar star
                
                if spectrum.I == spectrum.NI:
                    to_out = f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo"
                    to_out_length = len(to_out)
                    self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo \t\t\t|         |\\\\\n", to_out_length)                
                return
            elif mg <= -0.005:
                logging.info("No, it is not a Lambda Boo")
            
            if spectrum.flag_LB == 2:
                logging.info("It appears to be a metal-weak star")
                spectrum.SPT = spectrum.hSPT = code2spt(spectrum.h_type)
                spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                spectrum.LUM = code2lum(spectrum.lum_code)
                
                logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak")
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak"
                spectrum.Iter[spectrum.I].chi2 = 1. # fake chi2 for peculiar star
                
                if spectrum.I == spectrum.NI:
                    to_out = f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak"
                    to_out_length = len(to_out)
                    self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} {spectrum.LUM} metal-weak\t\t\t|         | {spectrum.note}\n", to_out_length)
                return
            elif mg <= -0.005:
                logging.info("No, it is not a metal-weak star")
            
            if spectrum.flag_LB == 3:
                logging.info("This star is probably rapidly rotating")
                flag_vsini = 1
        
            if spectrum.flag_LB == 0:
                logging.info("This is not a metal-weak star")
            
            flag_early = False
            spt_h = brent(spectrum.spt - 1, spectrum.spt, hydrogen_profile, args=(library, spectrum))
            logging.info(f"Hydrogen-line type = {spt_h:.6f}")
            
            # Determine the K-line type
            spt_K = brent(spectrum.spt - 1, spectrum.spt, spt_CaK, args=(library, spectrum))
            logging.info(f"K-line type = {spt_K:.6f}")
            
            # Pass it back to class_FG if it is F-type but not an Am star #
            if spt_K > 22.8 and spt_h > 22.8 and spectrum.iterate < 5:
                spectrum.spt = spt_h
                self.class_FG(spectrum)
                return
            
            # Determine metallic-line type, assuming preliminary luminosity type #
            spt_m = brent(spectrum.spt - 1, spectrum.spt, spt_metal, args=(library, spectrum))
            logging.info(f"Metallic-line type = {spt_m:.6f}")
            
            # If the resolution is poor, the metallic-line type can be faulty, so if spt_m < spt_K, set spt_m = spt_K #
            if spt_m < spt_K - 2:
                spt_m = spt_K
                logging.info("Correcting metallic-line type to K-line type")
            
            # Determine K-line type, assuming preliminary luminosity type #
            spt_K = brent(spectrum.spt - 1, spectrum.spt, spt_CaK, args=(library, spectrum))
            logging.info(f"K-line type = {spt_K:.6f}")
            logging.info(f"Metallic-line type = {spt_m:.6f}")
            logging.info(f"Hydrogen-line type = {spt_h:.6f}")
            Am_flag = 0
            if spt_m - spt_K >= 2.: Am_flag = 1
            elif (spt_m + spt_K) / 2 < spt_h - 2: Am_flag = 2
            logging.info(f"Am_flag = {Am_flag}")
        
        if Am_flag == 0:
            # Take a mean temperature type. For early A-type stars, overweight the metallic and K-line types. #
            if spectrum.spt <= 19.5:
                spectrum.sp_code = spectrum.spt = (spt_m + 3.0 * spt_K) / 4.
            else:
                spectrum.sp_code = spectrum.spt = (spt_h + spt_m + spt_K) / 3.
            logging.info(f"Merged spt = {spectrum.spt:.6f}")
            
            if spectrum.spt < 15. and spectrum.iterate < 5:
                self.class_B(spectrum)
                return
            
            # Now improve the luminosity type #
            spectrum.lum = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
            spectrum.lum = min(abs(spectrum.lum), 5.2)
            logging.info(f"lum = {spectrum.lum:.6f}")
            
            spectrum.pec = peculiarity(library, spectrum)
            if spectrum.pec:
                spectrum.PEC = " "
                if spectrum.Sr == 1: spectrum.PEC += "(Sr)"
                elif spectrum.Sr == 2: spectrum.PEC += "Sr"
                if spectrum.Si == 1: spectrum.PEC += "Si"
                if spectrum.Eu == 1: spectrum.PEC += "Eu"
                if spectrum.Cr == 1: spectrum.PEC += "Cr"
            
            spectrum.lum_code = spectrum.lum
            
            # Introduce chi2 selection code starting here
            spectrum.SPT = code2spt(spectrum.spt)
            spectrum.LUM = code2lum(spectrum.lum)
            CHI2 = match(library, spectrum)
            spectrum.qual = quality(spectrum, CHI2)
            spectrum.Iter[spectrum.I].chi2 = CHI2
            if not spectrum.pec:
                if spectrum.I <= spectrum.NI:
                    if flag_vsini == 1: spectrum.LUM += 'n'
                    to_out = f"{spectrum.SPT} {spectrum.LUM}"
                    to_out_length = len(to_out)
                    if to_out_length <= 9:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t\t\t{spectrum.qual}"
                    elif 9 < to_out_length <= 17:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t\t{spectrum.qual}"
                    else:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} \t{spectrum.qual}"
                
                if spectrum.I == spectrum.NI:
                    # Suppress hypergiants as code is not competent
                    if spectrum.lum < 0.:
                        self.write_to_out("Unclassifiable             |         | \\\\ \n")
                    else:
                        J = find_best(spectrum.Iter, spectrum.NI)
                        self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                        logging.info(f"\nBest iteration: I = {J}")
                        spectrum.done = True
                
                logging.info(f"{spectrum.I}: {spectrum.SPT} {spectrum.LUM} {CHI2:7.1e}")
            else:
                if spectrum.I <= spectrum.NI:
                    if flag_vsini == 1: spectrum.LUM += 'n'
                    to_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC}"
                    to_out_length = len(to_out)
                    if to_out_length <= 9:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t\t{spectrum.qual}"
                    elif 9 < to_out_length <= 17:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t{spectrum.qual}"
                    else:
                        spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t{spectrum.qual}"
                
                if spectrum.I == spectrum.NI:
                    if spectrum.lum < 0.:
                        self.write_to_out("Unclassifiable             |         | \\\\ \n")
                    else:
                        J = find_best(spectrum.Iter, spectrum.NI)
                        self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                        logging.info(f"\nBest iteration: I = {J}")
                        spectrum.done = True
                
                logging.info(f"{spectrum.I}: {spectrum.SPT} {spectrum.LUM} {spectrum.PEC} {CHI2:7.1e}")
        elif Am_flag == 1:
            spectrum.kSPT = code2spt(spt_K)
            spectrum.hSPT = code2spt(spt_h)
            spectrum.mSPT = spectrum.SPT = code2spt(spt_m)
            spectrum.spt = spt_h
            spectrum.PEC = " "
            spectrum.pec = peculiarity(library, spectrum)
            if spectrum.pec:
                spectrum.PEC = " "
                if spectrum.Sr == 1: spectrum.PEC += "Sr"
                if spectrum.Si == 1: spectrum.PEC += "Si"
                if spectrum.Eu == 1: spectrum.PEC += "Eu"
                if spectrum.Cr == 1: spectrum.PEC += "Cr"
            
            logging.info(f"{spectrum.I}: k{spectrum.kSPT}h{spectrum.hSPT}m{spectrum.mSPT} {spectrum.PEC}")
            spectrum.Iter[spectrum.I].format_out = f"k{spectrum.kSPT}h{spectrum.hSPT}m{spectrum.mSPT} {spectrum.PEC}"
            
            if spectrum.I == spectrum.NI:
                to_out = f"k{spectrum.kSPT}h{spectrum.hSPT}m{spectrum.mSPT} {spectrum.PEC}"
                to_out_length = len(to_out)
                self.write_to_out(f"k{spectrum.kSPT}h{spectrum.hSPT}m{spectrum.mSPT} {spectrum.PEC}\t\t\t|         |{spectrum.note}\n", to_out_length)
                spectrum.done = True
        elif Am_flag == 2:
            spectrum.mSPT = code2spt(spt_m)
            spectrum.hSPT = spectrum.SPT = code2spt(spt_h)
            spectrum.spt = spt_h
            
            logging.info(f"{spectrum.I}: {spectrum.hSPT} m{spectrum.mSPT} metal weak")
            spectrum.Iter[spectrum.I].chi2 = 1.
            spectrum.Iter[spectrum.I].format_out = f"{spectrum.hSPT} m{spectrum.mSPT} metal weak"
            
            if spectrum.I == spectrum.NI:
                to_out = f"{spectrum.hSPT} m{spectrum.mSPT} metal weak"
                to_out_length = len(to_out)
                self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} metal weak\t\t\t|         | {spectrum.note}\n", to_out_length)
                spectrum.done = True
        else:
            if spectrum.I == spectrum.NI:
                self.write_to_out("Undetermined peculiar A star\t|         |\\\\\n")
                spectrum.done = True
            logging.info(f"{spectrum.I}: Undetermined peculiar A-type star")
        
        return

    def class_FG(self, spectrum: Spectrum) -> None:
        """
        Classifies stars as F- and G-type stars based on their spectrum data.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object containing the star's spectral data, including wavelength, flux, preliminary spectral type, luminosity, and other relevant attributes.
        """
        library = self.library
        spectrum.PEC = " "
        logging.info("\nClassifying this star as an F/G star")
        spectrum.iterate += 1
        spectrum.sf = 3
        
        lum_orig = spectrum.lum
        spt_orig = spectrum.spt
        
        # Determine the hydrogen-line type
        spt_h = brent(spectrum.spt - 1, spectrum.spt, hydrogen_index, args=(library, spectrum))
        spectrum.SPT = code2spt(spt_h)
        logging.info(f"Hydrogen-line type = {spectrum.SPT} {spt_h:.6f}")
        
        # If it is an A-type star, pass it back to the A module
        if spt_h <= 22.8 and spectrum.iterate < 5:
            spectrum.spt = spt_h
            self.class_A(spectrum)
            return
        
        # If it is a K-type star, pass it to the KM module
        if spt_h >= 34. and spectrum.iterate < 5:
            spectrum.spt = min(spt_h, 36.)
            self.class_KM(spectrum)
            return
        
        # Determine the hydrogen to metal spectral type
        spt_hm = spt_h
        if spt_h >= 30.:
            spt_hm = brent(spt_h - 1, spt_h, spt_G_lines, args=(library, spectrum))
            spectrum.SPT = code2spt(spt_hm)
            logging.info(f"Hydrogen to metallic line type = {spectrum.SPT} {spt_hm:.6f}")
        
        spt_m = brent(spt_h - 1, spt_h, spt_metal, args=(library, spectrum))
        spectrum.SPT = code2spt(spt_m)
        logging.info(f"Metallic-line type = {spectrum.SPT} {spt_m:.6f}\n")
        
        # If hydrogen-line type and metallic line type differ too widely, set a peculiarity flag for future reference #
        flag_metal = False
        flag_rich = 0
        if abs(spt_m - spt_h) > 2. or abs(spt_hm - spt_h) > 3.: flag_metal = True
        if spt_m - spt_h > 1.: flag_rich = 1
        if spt_m - spt_h > 2.: flag_rich = 2
        if flag_rich != 0: logging.info("Star metal rich?")
        
        logging.info(f"flag_metal = {flag_metal}")
        
        # If things look normal, try to determine a luminosity type #
        if not flag_metal:
            spectrum.spt = (3. * spt_h + 2. * spt_hm + spt_m) / 6.
            logging.info(f"Normal star: spt = {spectrum.spt:.6f}")
            spectrum.sp_code = spectrum.spt
            spectrum.Ba = False # So that all luminosity criteria are considered
            spectrum.lum = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
            logging.info(f"lum = {spectrum.lum:.6f}")
            spectrum.lum = min(abs(spectrum.lum), 5.)
            spectrum.SPT = code2spt(spectrum.spt)
            spectrum.LUM = code2lum(spectrum.lum)
        
        ######################################
        # Now, begin check for peculiarities #
        ######################################
        
        # If Hydrogen-line type is earlier than F1, check to see if Lambda Boo #
        if spt_h <= 23.5 and flag_metal:
            mg = MgII(library, spectrum)
            logging.info(f"mg = {mg:.6f}")
            if mg <= -0.005:
                logging.info("This looks like it could be a metal-weak star")
                spectrum.flag_LB = lam_boo_2(library, spectrum)
                if spectrum.flag_LB:
                    logging.info("It appears to be a Lambda Boo star")
                    spectrum.hSPT = code2spt(spectrum.h_type)
                    spectrum.SPT = spectrum.mSPT = code2spt(spectrum.metal_type)
                    logging.info(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo")
                    if spectrum.I == spectrum.NI:
                        self.write_to_out(f"{spectrum.hSPT} m{spectrum.mSPT} V Lam Boo\n")
                    return
        
        # If Hydrogen-line type is earlier than F5, check for Am characteristics #
        if spt_h <= 27.:
            spt_K = brent(spt_h - 1, spt_h, spt_CaK, args=(library, spectrum))
            logging.info(f"spt_K = {spt_K:.6f} spt_m = {spt_m:.6f}")
            
            """
            If K-line type is earlier than metallic-line type, and the metallic type is equal to or later than the hydrogen-line type, it is probably an Am star, so pass to `class_A` function.
            """
            if spt_m - spt_K >= 2. and spt_m - spt_h >= -0.5 and spectrum.iterate < 5:
                spectrum.spt = spt_h
                self.class_A(spectrum)
                return
            
            # Also check for Ap-type peculiarities
            spectrum_ = copy.deepcopy(spectrum)
            spectrum_.spt = spectrum_.sp_code = spt_h
            spectrum.pec = peculiarity(library, spectrum_)
            if spectrum.pec:
                spectrum.PEC = " "
                if spectrum.Sr == 1: spectrum.PEC += "(Sr)"
                if spectrum.Sr == 2: spectrum.PEC += "Sr"
                if spectrum.Si == 1: spectrum.PEC += "Si"
                if spectrum.Cr == 1: spectrum.PEC += "Cr"
            
        # Determine an approximate luminosity class, leaving out Sr II lines. Then, check for a barium peculiarity. #
        spectrum.sp_code = spt_h
        spectrum.Ba = True # So that Sr II is left out of luminosity classification
        lum_Ba = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
        logging.info(f"lum_Ba = {lum_Ba:.6f}")
        lum_Ba = min(abs(lum_Ba), 5.)
        
        # The barium function checks for Sr peculiarity if earlier than G5, for a Ba peculiarity if G5 or later.
        # Check for Ba (Sr) peculiarity #
        spectrum_ = copy.deepcopy(spectrum)
        spectrum_.lum = spectrum_.lum_code = lum_Ba
        spectrum.Ba = barium(library, spectrum_)
        
        # Check for Carbon peculiarities #        
        logging.info(f"C2 ratio = {carbon_4737(library, spectrum):.6f}")
        logging.info(f"CN ratio = {CN_4215(library, spectrum):.6f}")
        C2 = CN = CH = 0.0
        if spectrum.lum < 4.5 and spectrum.spt < 39.0:
            C2 = carbon_4737(library, spectrum)
            CN = CN_4215(library, spectrum)
        
        if 32. <= spectrum.spt < 39. and spectrum.lum < 4.5:
            CH = CH_band2(spectrum.spt, spectrum.lum, library, spectrum)
            logging.info(f"CH ratio = {CH:.6f}")
        
        if 0.025 < abs(CN) < 0.05:
            spectrum.pec = True
        elif 0.05 <= CN < 0.075:
            spectrum.pec = True
            spectrum.PEC += " CN1"
        elif CN >= 0.075:
            spectrum.pec = True
            spectrum.PEC += " CN2"
        elif CN <= -0.05:
            spectrum.pec = True
            spectrum.PEC += " CN-1"
        
        if spectrum.spt >= 32.:
            if 0.075 < CH < 0.15:
                spectrum.pec = True
            elif 0.15 <= CH < 0.2:
                spectrum.pec = True
                spectrum.PEC += " CH1"
            elif CH > 0.2:
                spectrum.pec = True
                spectrum.PEC += " CH2"
            elif -0.15 < CH <= -0.075:
                spectrum.pec = True
            elif -0.2 < CH <= -0.15:
                spectrum.pec = True
                spectrum.PEC += " CH-1"
            elif CH <= -0.2:
                spectrum.pec = True
                spectrum.PEC += " CH-2"
        
        if spectrum.Ba:
            spectrum.pec = True
            spectrum.lum = lum_Ba
            spectrum.PEC += (" Sr" if spectrum.spt < 32. else " Ba")
        
        spectrum.SPT = code2spt(spectrum.spt)
        spectrum.LUM = code2lum(spectrum.lum)
        
        spectrum.sp_code = spectrum.spt
        spectrum.lum_code = spectrum.lum
        
        # If Ba is not peculiar, then accept the earlier luminosity class #
        sub_h = sub_class(spt_h)
        sub_m = sub_class(spt_m)
                
        fe = 0.
        if sub_h - sub_m > 2.: fe = -0.13 * (sub_h - sub_m) - 0.26
        if sub_m - sub_h > 2.: fe = 0.25 * (sub_m - sub_h)
        if sub_m - sub_h > 1. and sub_m - sub_h < 2.: flag_rich = 1

        if abs((sub_h - sub_m) < 2. or abs(fe) < 0.5):
            CHI2 = match(library, spectrum)
            spectrum.qual = quality(spectrum, CHI2)
            if spectrum.I <= spectrum.NI:
                spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM}"
                spectrum.Iter[spectrum.I].chi2 = CHI2
                to_out = f"{spectrum.SPT} {spectrum.LUM}"
                to_out_length = len(to_out)

            logging.info(f"{spectrum.I}: {spectrum.SPT} {spectrum.LUM}  {CHI2:7.1e}")
            if flag_rich == 1: logging.info(" ((metal-rich))")
            
            if not spectrum.pec:
                if spectrum.I <= spectrum.NI:
                    spectrum.Iter[spectrum.I].format_out += f" \t\t\t{spectrum.qual}"
                    
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                    logging.info(f"\nBest iteration: I = {J}")
                    spectrum.done = True
            
            else:
                if spectrum.I <= spectrum.NI:
                    to_out_length += len(spectrum.PEC)
                    if to_out_length <= 9:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t\t\t{spectrum.qual}"
                    elif 9 < to_out_length <= 17:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t\t{spectrum.qual}"
                    else:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t{spectrum.qual}"
                    
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                    logging.info(f"\nBest iteration: I = {J}")
                    spectrum.done = True
                
                logging.info(f"{spectrum.PEC} {CHI2:7.1e}")    
        else:
            # metal-weak or metal-rich stars #
            spectrum.SPT = code2spt(spt_h)
            spectrum.sp_code = spectrum.spt = spt_h
            logging.info(f"Metal weak or rich: spt = {spectrum.spt:.6f}")
            
            spectrum.lum = brent(2.5, 3.0, lum_ratio_min, args=(library, spectrum))
            
            if abs(spectrum.lum) > 5.0:
                spectrum.lum = spectrum.lum_code = 5.0
                spt_m = brent(spt_h - 1, spt_h, spt_metal, args=(library, spectrum))
                spectrum.SPT = code2spt(spt_m)
                sub_h = sub_class(spt_h)
                sub_m = sub_class(spt_m)
                fe = -0.13 * (sub_h - sub_m) - 0.26
            
            spectrum.lum_code = spectrum.lum
            spectrum.SPT = code2spt(spt_h)
            spectrum.LUM = code2lum(spectrum.lum)
            
            spectrum_ = copy.deepcopy(spectrum)
            spectrum_.spt = spt_h
            CHI2 = match(library, spectrum_)
            spectrum.Iter[spectrum.I].chi2 = CHI2
            spectrum.qual = quality(spectrum_, CHI2)
            
            flag_fe = 0
            if abs(fe) > 0.5:
                if spectrum.I <= spectrum.NI:
                    to_out = spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} Fe{fe:+4.1f}"
                    to_out_length = len(to_out)
                
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    spectrum.done = True
                
                logging.info(f"{spectrum.I}:  {spectrum.SPT} {spectrum.LUM} Fe{fe:+4.1f}")
                flag_fe = 1
            else:
                if spectrum.I <= spectrum.NI:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} "
                
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    to_out = f"{spectrum.SPT} {spectrum.LUM} "
                    to_out_length = len(to_out)
                    spectrum.done = True
                
                logging.info(f"{spectrum.I}:   {spectrum.SPT} {spectrum.LUM} ")
            
            if not spectrum.pec and flag_fe == 0:
                if spectrum.I <= spectrum.NI:
                    to_out_length = len(spectrum.Iter[spectrum.I].format_out)
                    if to_out_length <= 9:
                        spectrum.Iter[spectrum.I].format_out += f"\t\t\t{spectrum.qual}"
                    elif 9 < to_out_length <= 17:
                        spectrum.Iter[spectrum.I].format_out += f"\t\t{spectrum.qual}"
                    else:
                        spectrum.Iter[spectrum.I].format_out += f"\t{spectrum.qual}"
                
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                    logging.info(f"\nBest iteration: I = {J}")
                    spectrum.done = True
            else:
                if spectrum.I <= spectrum.NI:
                    to_out_length = len(spectrum.Iter[spectrum.I].format_out)
                    to_out_length += len(spectrum.PEC)
                    if to_out_length <= 9:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t\t\t{spectrum.qual}"
                    elif 9 < to_out_length <= 17:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t\t{spectrum.qual}"
                    else:
                        spectrum.Iter[spectrum.I].format_out += f"{spectrum.PEC} \t{spectrum.qual}"
                
                if spectrum.I == spectrum.NI:
                    J = find_best(spectrum.Iter, spectrum.NI)
                    self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                    logging.info(f"\nBest iteration: I = {J}")
                    spectrum.done = True
                
                logging.info(f"{spectrum.PEC} {CHI2:7.1e}")
        
        return
        
    def class_KM(self, spectrum: Spectrum) -> None:
        """
        Classifies stars as K- and M-type stars based on their spectrum data.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum object containing the star's spectral data, including wavelength, flux, preliminary spectral type, luminosity, and other relevant attributes.
        """        
        library = self.library
        spectrum.PEC = ""
        
        # Iterate on temperature and luminosity types twice #
        logging.info("\nClassifying this star as a K/M star")
        spectrum.iterate += 1
        
        """
        First look at the hydrogen-line spectral type; hydrogen lines don't constitute a good criterion for K/M stars, but can exclude a G spectral type.
        NOTE: hydrogen index will give a latest hydrogen-line type of K2.
        """
        spt_h = brent(spectrum.spt - 1, spectrum.spt, hydrogen_index, args=(library, spectrum))
        spectrum.SPT = code2spt(spt_h)
        logging.info(f"{spectrum.I}: Hydrogen-line type = {spectrum.SPT}")
        
        """
        Now the metallic-line type. Again, not great for K/M stars, but this can help detect an early K, metal-weak star.
        """
        spt_m = brent(spt_h - 1, spt_h, spt_metal, args=(library, spectrum))
        spectrum.SPT = code2spt(spt_m)
        logging.info(f"{spectrum.I}: Metallic-line type = {spectrum.SPT}\n")
        
        if spt_h < 33.0:
            spectrum.spt = spt_h
            self.class_FG(spectrum)
            return
        
        """
        If the following is satisfied, the star is a metal-weak early K-type star.
        CAUTION: Application of this criterion beyond K5 will pick up the natural decline in blue-violet line strengths in late K and M stars because of increased violet opacity.
        """
        flag_metal = False
        if 34.0 <= spt_h <= 36.0 and spt_m < 32.5:
            flag_metal = True
        
        logging.info(f"flag_metal = {flag_metal}")
        # Check to see if there is a Barium peculiarity
        spectrum.Ba = barium(library, spectrum)
        
        # If star is normal, iterate on temperature and luminosity classifications
        if not flag_metal:
            spt_km = spectrum.spt
            spectrum.sp_code = spt_km
            lum_km = spectrum.lum
            spectrum.lum_code = lum_km
            
            spt_km = brent(spt_km - 1, spt_km, spt_KM, args=(library, spectrum))
                        
            if spt_km > library.s_cool:
                spt_km = library.s_cool
                logging.info("Caution, spectral type may be cooler than library limit")
            
            spectrum.sp_code = spectrum.spt = spt_km
            spectrum.SPT = code2spt(spectrum.spt)
            logging.info(f"{spectrum.I}: spt_km = {spectrum.spt:.6f}")
            lum_km = brent(lum_km, lum_km + 0.5, lum_ratio_min, args=(library, spectrum))

            if abs(lum_km) > 5.2: lum_km = spectrum.lum_code = 5.2
            # Because of standard library deficiency (no supergiants after M2), if lum < 3.0 for stars later than M2, set lum = 3.0 #
            if spectrum.spt > 42.5 and lum_km < 3.0: lum_km = 3.0
            # No dwarf standards later than library limit, so adjust everything to giant
            if spectrum.spt >= library.s_cool and lum_km > 3.0: lum_km = 3.0
            spectrum.lum_code = spectrum.lum = lum_km
            spectrum.LUM = code2lum(spectrum.lum)
            
            # Check on carbon peculiarities #
            logging.info(f"C2 ratio = {carbon_4737(library, spectrum):.6f}")
            logging.info(f"CN ratio = {CN_4215(library, spectrum):.6f}")
            C2 = CN = CH = 0.0
            if spectrum.spt < 39. and spectrum.lum < 4.5:
                C2 = carbon_4737(library, spectrum)
                CN = CN_4215(library, spectrum)
                CH = CH_band2(spectrum.spt, spectrum.lum, library, spectrum)
            
            logging.info(f"CH ratio = {CH:.6f}")
            if 0.05 <= CN < 0.075: spectrum.PEC += " CN1"
            elif CN >= 0.075: spectrum.PEC += " CN2"
            elif CN <= -0.05: spectrum.PEC += " CN-1"
            
            if 0.15 <= CH < 0.2: spectrum.PEC += " CH1"
            elif CH >= 0.2: spectrum.PEC += " CH2"
            elif -0.2 < CH <= -0.15: spectrum.PEC += " CH-1"
            elif CH <= -0.2: spectrum.PEC += " CH-2"
            
            spectrum.Ba = barium(library, spectrum)
            # Suppress Ba peculiarity for M stars
            if spectrum.Ba and spectrum.spt < 40.0: spectrum.PEC += " Ba"
            logging.info(f"{spectrum.I}:  lum_km = {lum_km:.6f}")
            
            if spt_km < 33. and spectrum.iterate < 5:
                spectrum.spt = spectrum.sp_code = spt_km
                spectrum.lum = spectrum.lum_code = lum_km
                self.class_FG(spectrum)
                return
            
            # Slight tendency to classify K and M dwarfs as IV-V
            if spectrum.I == spectrum.NI and spectrum.lum > 4.5: spectrum.lum = 5.0
            spectrum.sp_code = spectrum.spt
            spectrum.lum_code = spectrum.lum
            spectrum.LUM = code2lum(spectrum.lum)
            spectrum.SPT = code2spt(spectrum.spt)
            CHI2 = match(library, spectrum)
            spectrum.qual = quality(spectrum, CHI2)
            
            to_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC}"
            to_out_length = len(to_out)
            if spectrum.I <= spectrum.NI:
                if to_out_length <= 9:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t\t{spectrum.qual}"
                elif 9 < to_out_length <= 17:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t\t{spectrum.qual}"
                else:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} {spectrum.PEC} \t{spectrum.qual}"
                
                spectrum.Iter[spectrum.I].chi2 = CHI2
            
            if spectrum.I == spectrum.NI:
                J = find_best(spectrum.Iter, spectrum.NI)
                self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                logging.info(f"\nBest iteration: I = {J}")
                spectrum.done = True
            
            logging.info(f"{spectrum.I}:  {spectrum.SPT} {spectrum.LUM} {spectrum.PEC} {CHI2:7.1e}")
        else:
            # Here we deal with metal-weak early K-type stars
            spectrum.SPT = code2spt(spt_h)
            lum_km = brent(spectrum.lum - 0.5, spectrum.lum, lum_ratio_min, args=(library, spectrum))
            if abs(lum_km) > 5.2: lum_km = spectrum.lum_code = 5.2
            spectrum.LUM = code2lum(lum_km)
            spectrum.lum_code = spectrum.lum = lum_km
            CHI2 = match(library, spectrum)
            spectrum.qual = quality(spectrum, CHI2)
            to_out = f"{spectrum.SPT} {spectrum.LUM} metal-weak"
            to_out_length = len(to_out)
            if spectrum.I <= spectrum.NI:
                if to_out_length <= 9:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} metal-weak \t\t\t{spectrum.qual}"
                elif 9 < to_out_length <= 17:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} metal-weak \t\t{spectrum.qual}"
                else:
                    spectrum.Iter[spectrum.I].format_out = f"{spectrum.SPT} {spectrum.LUM} metal-weak \t{spectrum.qual}"
                
                spectrum.Iter[spectrum.I].chi2 = CHI2
            if spectrum.I == spectrum.NI:
                J = find_best(spectrum.Iter, spectrum.NI)
                self.write_to_out(f"{spectrum.Iter[J].format_out}\n")
                logging.info(f"\nBest iteration: I = {J}")
                spectrum.done = True
            
            logging.info(f"{spectrum.I}:  {spectrum.SPT} {spectrum.LUM} metal-weak {CHI2:7.1e}")
        
        logging.info("KM return")
        return

    def write_to_out(self, s: str, s_len = 0) -> None:
        """
        Appends a string to an output file with conditional formatting based on the length of the string.

        If the output file is not specified (i.e., `self.out_file` is `None`), the method does nothing.

        Parameters
        ----------
        s : str
            The string to be appended to the output file.
            
        s_len : int, optional
            The length of the string `s`. If not provided, it defaults to 0. 
            This parameter determines the formatting applied when writing the string to the file.
        """
        if self.out_file == None: return
        if s_len == 0:
            with open(self.out_file, 'a') as f: f.write(s)
            return
        else:
            import re
            with open(self.out_file, 'a') as f:
                if s_len <= 9:
                    f.write(s)
                elif 9 < s_len <= 17:
                    f.write(re.sub(r'\t\t\t', '\t\t', s))
                else:
                    f.write(re.sub(r'\t\t\t', '\t', s))
            return

def rough_type_1(library: Library, spectrum: Spectrum) -> None:
    """
    Supplies a rough initial spectral type for rectified spectra.

    Parameters:
    ----------
    library : Library object
        An object containing the library of standard spectra and relevant parameters.

    spectrum : Spectrum object
        An object representing the input spectrum. 

    Returns:
    -------
    None
        The function modifies the `spectrum` object in place, setting the `isp` and `ilt`
        attributes to the best-matching spectral type and luminosity class, respectively.
    """
    # File names for different spectral types
    SD = ["t030l50p00.rbn","t070l50p00.rbn","t120l50p00.rbn","t190l50p00.rbn","t230l50p00.rbn", "t260l50p00.rbn","t320l50p00.rbn","t360l50p00.rbn","t400l50p00.rbn","t425l50p00.rbn"]
    SG = ["t030l30p00.rbn","t070l30p00.rbn","t120l30p00.rbn","t190l30p00.rbn","t230l30p00.rbn", "t260l30p00.rbn","t320l30p00.rbn","t360l30p00.rbn","t400l30p00.rbn","t425l30p00.rbn"]
    SS = ["t030l10p00.rbn","t070l10p00.rbn","t120l10p00.rbn","t190l10p00.rbn","t230l10p00.rbn", "t260l10p00.rbn","t320l10p00.rbn","t360l10p00.rbn","t400l10p00.rbn","t425l10p00.rbn"]

    # ISP and ILT arrays
    ISP = np.array([3.0, 7.0, 12.0, 19.0, 23.0, 26.0, 32.0, 36.0, 40.0, 42.5])
    ILT = np.array([5.0, 3.0, 1.0])
    chi = np.full((3, 10), np.inf)  # Initialize chi array with large values

    # Determine hot and cool indices
    i_hot = np.argmax(library.s_hot >= ISP) - 1
    i_cool = np.argmax(library.s_cool > ISP) - 1

    # Rebin spectrum
    _, flux_rebin, _, _ = rebin(spectrum.wave, spectrum.flux, library.w_low, library.w_high, library.space)

    # Calculate chi values for each spectral type and luminosity class
    for l in range(3):
        for i in range(i_hot, i_cool + 1):
            chi[l, i] = 0.0
            file_name = os.path.join(library.MKLIB, library.name, [SD, SG, SS][l][i])
            with open(file_name, 'r') as in9:
                for j, line in enumerate(in9):
                    wave_temp, flux_temp = map(float, line.split())
                    if library.w_low + 100.0 <= wave_temp <= library.w_high - 100.0 and flux_temp != 0.0:
                        chi[l, i] += (flux_temp - flux_rebin[j]) ** 2

    # Find minimum chi value and corresponding spectral type and luminosity class
    min_index = np.unravel_index(np.argmin(chi), chi.shape)
    j, k = min_index[1], min_index[0]

    # Set output values
    spectrum.isp = ISP[j]
    spectrum.ilt = ILT[k]

def rough_type_2(library: Library, spectrum: Spectrum) -> None:
    """
    Determines the initial rough spectral type using various indices and ratios.
    
    Parameters:
    ----------
    library : Library object
        An object containing the library of standard spectra and relevant parameters.

    spectrum : Spectrum object
        An object representing the input spectrum.

    Returns:
    -------
    None
        The function modifies the `spectrum` object in place, setting the `isp` and `ilt`
        attributes to the best-matching spectral type and luminosity class, respectively.
    """
    t = np.array([9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 36.0, 37.0, 39.0, 40.0, 40.7, 42.5, 44.0, 45.5])

    # Rebin the spectrum
    wave_rebin, flux_rebin, _, _ = rebin(spectrum.wave, spectrum.flux, library.w_low, library.w_high, library.space)

    # Initialize variables
    indices = {
        "C1": 0.0,
        "CaK": 0.0,
        "C2": 0.0, 
        "Hdelta": 0.0, 
        "C3": 0.0, 
        "C4": 0.0,
        "HeI4026": 0.0, 
        "C5": 0.0, 
        "C6": 0.0,
        "CaI": 0.0, 
        "C7": 0.0, 
        "C8": 0.0,
        "Gband": 0.0, 
        "C9": 0.0, 
        "C10": 0.0,
        "HeI4471": 0.0, 
        "C11": 0.0, 
        "C12": 0.0,
        "C13": 0.0
    }
    
    # Calculate indices
    ranges = [(3918., 3925.), (3927., 3937.),
              (4022., 4052.), (4062., 4142.),
              (4152., 4182.), (4014., 4020.),
              (4020., 4032.), (4032., 4038.),
              (4210., 4221.), (4221., 4232.),
              (4232., 4243.), (4233., 4248.),
              (4297., 4314.), (4355., 4378.),
              (4452., 4462.), (4462., 4480.),
              (4480., 4490.), (4918., 4948.),
              (4958., 4988.)]
    index_keys = [
    "C1", "CaK", "C2", "Hdelta", "C3", "C4",
    "HeI4026", "C5", "C6", "CaI", "C7", "C8",
    "Gband", "C9", "C10", "HeI4471", "C11", "C12", "C13"
]
    for key, range in zip(index_keys[:-2], ranges[:-2]):
        mask = (wave_rebin >= range[0]) & (wave_rebin <= range[1])
        indices[key] += np.sum(flux_rebin[mask])

    if library.s_cool >= 40.0:
        for key, range in zip(index_keys[-2:], ranges[-2:]):
            mask = (wave_rebin >= range[0]) & (wave_rebin <= range[1])
            indices[key] += np.sum(flux_rebin[mask])

    C1 = indices["C1"] / 7
    CaK = indices["CaK"] / 10
    C2 = indices["C2"] / 30
    Hdelta = indices["Hdelta"] / 80
    C3 = indices["C3"] / 30
    C4 = indices["C4"] /6
    HeI4026 = indices["HeI4026"] / 12
    C5 = indices["C5"] / 6
    C6 = indices["C6"] / 11
    CaI = indices["CaI"] / 11
    C7 = indices["C7"] / 11
    C8 = indices["C8"] / 15
    Gband = indices["Gband"] / 17
    C9 = indices["C9"] / 23
    C10 = indices["C10"] / 10
    HeI4471 = indices["HeI4471"] / 12
    C11 = indices["C11"] / 10
    C12 = indices["C12"]
    C13 = indices["C13"]
    
    # Calculate ratios
    CaK /= C1
    Hdelta /= (C2 + C3)
    HeI4026 /= (C4 + C5)
    CaI /= (C6 + C7)
    Gband /= (0.484 * C8 + 0.516 * C9)
    HeI4471 /= (C10 + C11)
    if library.s_cool >= 40.0:
        TiO = C12 / C13
    
    logging.info(f"Gband = {Gband:1.6f} CaI = {CaI:1.6f} HeI4471 = {HeI4471:1.6f} CaK = {CaK:1.6f} TiO = {TiO:1.6f}")
    
    # Determine spectral type
    sp1 = GbandCaI(CaI, Gband)
    sp2 = CaKHe4471(CaK, HeI4471)
    sp3 = library.s_cool
    if library.s_cool >= 40.0:
        sp3 = TiOIndex(TiO)
    
    logging.info(f"Rough_type_2: sp1 = {sp1:.6f} sp2 = {sp2:.6f} sp3 = {sp3:.6f}")

    spt = 20.0
    if sp1 < 0: spt = sp3
    if 0 <= sp1 <= 20: spt = sp2
    if sp2 >= 20 or sp1 >= 20: spt = sp1
    if sp1 <= 0 and sp2 <= 0: spt = sp3
    if library.s_cool >= 40.0 and sp3 >= 40.0 and sp1 >= 38.0: spt = sp3
    if TiO > 1.10: spt = sp3  # Ensures M-type dwarfs are not missed
    if spt > library.s_cool: spt = library.s_cool - 2.0
    if spt < library.s_hot: spt = library.s_hot + 2.0
    
    logging.info(f"Rough_type_2: final spt = {spt:.6f}")

    # Find the closest spectral type
    close = 50.0
    j = 0
    for i, t_val in enumerate(t):
        if abs(spt - t_val) <= close:
            close = abs(spt - t_val)
            j = i

    spt = t[j]

    # Set output values
    spectrum.isp = spt
    spectrum.ilt = 5.0


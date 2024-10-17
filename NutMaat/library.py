import os

class Library:
    """
    A class to represent a spectral library and its associated parameters.

    This class initializes a spectral library by reading from a specified library file 
    (`nutmaat.lib`) to extract details about the spectral ranges and other relevant attributes.

    Parameters
    ----------
    name : str
        The name of the spectral library to be initialized.
        
    MKLIB : str
        The directory path where the spectral library file is located.

    Raises
    ------
    FileNotFoundError
        If the `nutmaat.lib` file cannot be found in the specified directory.
        
    Exception
        For any other exceptions that may occur while reading the library file.
    """
    def __init__(self, name: str, MKLIB: str) -> None:
        from .encoding import spt2code
        
        self.name = name
        self.MKLIB = MKLIB
        self.lib_path = os.path.join(MKLIB, self.name)
        
        # Get details of spectral library from nutmaat.lib
        self.w_low = 3800.0
        self.w_high = 4600.0
        self.space = 0.5
        self.flag_lib = False
        self.flag_template = False
        LIB = os.path.join(MKLIB, "nutmaat.lib")
        
        try:
            with open(LIB, "r") as mk:
                for line in mk:
                    _, Lib = line.split()[:2]
                    ni = mk.readline().split()
                    w_low, w_high, space = float(ni[1]), float(ni[2]), float(ni[3])
                    ni = mk.readline().split()
                    sp_low, sp_high = ni[1], ni[2]
                    prelim = mk.readline().split()[1].split(',')
                    _, template = mk.readline().split()
                    _ = mk.readline().strip()
                    if Lib == self.name:
                        self.flag_lib = True
                        self.w_low = w_low
                        self.w_high = w_high
                        self.space = space
                        self.sp_low =  sp_low
                        self.sp_high = sp_high
                        self.prelim = prelim
                        self.template = template
                        
                        if self.template == 'yes': self.flag_template = True
                        self.s_hot = spt2code(self.sp_low)
                        self.s_cool = spt2code(self.sp_high)
                        break
        except FileNotFoundError:
            print(f"Cannot find the file {LIB}")
        except Exception as e:
            print(f"An error occurred: {e}")
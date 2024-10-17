"""
Example of smoothing one spectrum using a data frame
"""

import os
import pandas as pd 
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier(out_file='output')

# defining the file_path and name
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', 'cflib_spec_1.dat'))
name = os.path.splitext(os.path.basename(file_path))[0]

# read the file into two separate arrays
data = pd.read_csv(file_path, sep='\s+', header=None)

# separate wavelengths and fluxes into arrays
wave = data[0].to_numpy()  # First column
flux = data[1].to_numpy()  # Second column

# create a Series with three columns: name, wave, flux
df = pd.Series({
    'name': name,
    'wave': wave,
    'flux': flux
})

# applying rebinning and smoothing methods
result = clf.srebin_spectrum(0.2, from_df=True, df=df, cols=df.index.tolist())
result = clf.smooth_spectrum(0.2, 1.0, 3.46, from_file=True, file_name=file_path)

# printing the result
print(result)
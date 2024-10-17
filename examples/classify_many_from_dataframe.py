"""
Example of classifying several spectra using a data frame
"""

import os
import pandas as pd 
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier(out_file='output')

# defining the file paths
file_names = []
for i in range(4):
    file_names.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', f'miles_spec_{i+1}.dat')))

# read the files into a dataframe
spectra = []
for file_name in file_names:
    data = pd.read_csv(file_name, sep='\s+', header=None)

    # separate wavelengths and fluxes into arrays
    wave = data[0].to_numpy()  # First column
    flux = data[1].to_numpy()  # Second column

    # create a data frame with three columns: name, wave, flux
    name = os.path.splitext(os.path.basename(file_name))[0]
    spectra.append(pd.Series({
        'name': name,
        'wave': wave,
        'flux': flux
    }))

df = pd.concat(spectra, ignore_index=True)

# applying classification method
result = clf.classify_spectra(2, 3, from_df=True, df=df, cols=df.index.tolist())

# printing the result
print(result)
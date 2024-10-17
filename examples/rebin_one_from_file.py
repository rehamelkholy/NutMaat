"""
Example of rebinning one spectrum using a data file
"""

import os
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier(out_file='output')

# defining the file_path
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', 'cflib_spec_1.dat'))

# applying rebinning method
result = clf.srebin_spectrum(0.2, from_file=True, file_name=file_path)

# printing the result
print(result)
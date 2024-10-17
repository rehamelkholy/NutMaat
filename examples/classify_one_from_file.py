"""
Example of classifying one spectrum using a data file
"""

import os
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier(out_file='output')

# defining the file_path
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', 'miles_spec_1.dat'))

# applying classification method
result = clf.classify_spectrum(2, 3, from_file=True, file_name=file_path)

# printing the result
print(result)
"""
Example of using a custom library
"""

import os
from NutMaat.classifier import Classifier

# defining the custom library path
lib_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'custom_lib')

# initializing the classifier
clf = Classifier(lib='my_lib', MKLIB=lib_path)

# defining the file_path
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', 'miles_spec_1.dat'))

# applying classification method
result = clf.classify_spectrum(2, 3, from_file=True, file_name=file_path)

# printing the result
print(result)
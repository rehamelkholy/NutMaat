"""
Example of classifying several spectra using data files
"""

import os
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier(out_file='output')

# defining the file paths
file_names = []
for i in range(4):
    file_names.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'spectra', f'miles_spec_{i+1}.dat')))

# applying classification method
result = clf.classify_spectra(2, 3, from_file=True, file_names=file_names)

# printing the result
print(result)
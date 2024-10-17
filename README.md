<p align="center">
    <img src=".\data\NutMaat.jpg" alt="NutMaat" style="width: 400px;">
</p>

# NutMaat

![GitHub Created At](https://img.shields.io/github/created-at/rehamelkholy/NutMaat) ![GitHub License](https://img.shields.io/github/license/rehamelkholy/NutMaat) ![GitHub last commit](https://img.shields.io/github/last-commit/rehamelkholy/NutMaat) ![GitHub repo size](https://img.shields.io/github/repo-size/rehamelkholy/NutMaat) ![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/rehamelkholy/NutMaat) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NutMaat) ![Maintenance](https://img.shields.io/maintenance/yes/2024) ![PyPI - Version](https://img.shields.io/pypi/v/NutMaat)

![GitHub watchers](https://img.shields.io/github/watchers/rehamelkholy/NutMaat) ![GitHub Repo stars](https://img.shields.io/github/stars/rehamelkholy/NutMaat) ![GitHub forks](https://img.shields.io/github/forks/rehamelkholy/NutMaat)


**NutMaat** is a Python package designed to classify stellar spectra on the MK Spectral Classification system in a way similar to humans—by direct comparison with the MK classification standards, based on the `MKCLASS` C package. The package is OS-independent, installable via `pip`, and integrated with to work with `pandas` data frames.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Citing NutMaat](#citing-nutmaat)
- [Contact](#contact)
- [License](#license)
  
## Features

- Spectral type determination by comparison to a standard library
- Detecting a number of spectral peculiarities, *e.g.* barium stars, carbon-rich giants, Am stars, etc.
- Evaluating the quality of the classification
- Capablity of classifying spectra in the violet–green region in either the rectified or flux-calibrated format
- No need to correct for reddening
- Possibility to use custom standard libraries
- Batching large number of stars for classification

<p align="center">
    <img src=".\data\spt_fit.png" alt="spectral type results" style="width: 600px;">
</p>

<p align="center">
    <img src=".\data\lum_error.png" alt="luminosity class error histogram" style="width: 600px;">
</p>

## Installation

NutMaat requires:
- numpy>=2.0.0
- pandas==2.2.2
- scipy>=1.14.0
- tqdm>=4.66.5

To install NutMaat, simply use `pip`:

```bash
pip install NutMaat
```

Alternatively, you can install it directly from the source by cloning the repository:

```bash
git clone https://github.com/rehamelkholy/nutmaat.git
cd NutMaat
python setup.py install
```

## Usage

Here's a quick example showing how to use NutMaat to classify a stellar spectrum from file:

```python
from NutMaat.classifier import Classifier

# initializing the classifier
clf = Classifier()

# defining the file_path
file_path = <your_file_path>

# applying classification method
result = clf.classify_spectrum(2, 3, from_file=True, file_name=file_path)

# printing the result
print(result)
```

For more advanced use cases such as classifying using `pandas` data frames, classifying several spectra at a time, or rebinning or smoothing spectra prior to classification, refer to the [`examples`](examples/) folder.

## Documentation

NutMaat comes with comprehensive docstrings embedded within the code. You can easily access the documentation for any function or class by using Python’s built-in `help()` function. For example:

```python
from NutMaat.classifier import Classifier
help(Classifier.classify_spectrum)
```

## Citing NutMaat

If you use NutMaat in your research or publications, please cite it using the following BibTeX entry:

```bibtex
@misc{nutmaat2024,
  author = {{El-Kholy}, R.~I. and {Hayman}, Z.~M.},
  title  = {NutMaat: A Python library for classifying stellar spectra based on the MKCLASS package},
  year   = {2024},
  doi    = {10.5281/zenodo.13945430},
  url    = {https://github.com/rehamelkholy/NutMaat},
}
```
Since NutMaat is based upon the C package `MCCLASS`, please also cite the following paper:

```bibtex
@article{mcclass2014,
  author = {{Gray}, R.~O. and {Corbally}, C.~J.},
  title = "{An Expert Computer Program for Classifying Stars on the MK Spectral Classification System}",
  journal = {\aj},
  year = 2014,
  volume = {147},
  number = {4},
  pages = {80},
  doi = {10.1088/0004-6256/147/4/80},
}
```

Additionally, we appreciate it if you mention NutMaat in the acknowledgments section of your papers or reports.

## Contact

If you have any questions, feedback, or need assistance, feel free to reach out:

- Email: **[relkholy@sci.cu.edu.eg](mailto:relkholy@sci.cu.edu.eg)** or **[reham.elkholy@cu.edu.eg](mailto:reham.elkholy@cu.edu.eg)**
- GitHub Issues: **[https://github.com/rehamelkholy/NutMaat/issues](https://github.com/rehamelkholy/NutMaat/issues)**

## License

NutMaat is licensed under the **MIT** License. See the [`LICENSE`](LICENSE) file for more details.

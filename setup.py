import os
from setuptools import setup, find_packages

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'NutMaat', '_version.py')
    with open(version_file) as f:
        exec(f.read())
    return locals()['__version__']

setup(
    name='NutMaat',
    version=get_version(),
    packages=find_packages(),
    author='Reham El-Kholy, PhD',
    author_email='reham.elkholy@cu.edu.eg',
    description='A Python library for classifying stellar spectra based on the MKCLASS package',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rehamelkholy/NutMaat',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Beta'
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=2.0.0',
        'pandas==2.2.2',
        'scipy>=1.14.0',
        'tqdm>=4.66.5',
    ],
)
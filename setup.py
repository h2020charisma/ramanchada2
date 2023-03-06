from os import path
from setuptools import find_packages, setup
import ast

NAME = 'ramanchada2'

with open(path.join(path.dirname(__file__), 'src/ramanchada2/__init__.py')) as fd:
    VERSION = [expr.value.value
               for expr in ast.parse(fd.read()).body
               if (isinstance(expr, ast.Assign) and
                   len(expr.targets) == 1 and
                   expr.targets[0].id == '__version__')
               ][-1]

DESCRIPTION = 'Harmonising Raman Spectroscopy'
README_FILE = path.join(path.dirname(__file__), 'README.pypi')
LONG_DESCRIPTION = open(README_FILE, 'r', encoding='utf-8').read()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/h2020charisma/ramanchada2'
AUTHOR = 'IDEAconsult Ltd.'
AUTHOR_EMAIL = 'dev-charisma@ideaconsult.net'
LICENSE = 'MIT'

KEYWORDS = [
    'Raman',
    'spectroscopy',
]

PYTHON_REQUIRES = '>=3.9'

PACKAGES = find_packages(where='src')

PACKAGE_DIR = {'': 'src'}

PACKAGE_DATA = {'': ['auxiliary/**/*.txt']}

DATA_FILES = []

INSTALL_REQUIRES = [
        "h5py",
        "lmfit",
        "matplotlib",
        "numpy",
        "pandas",
        "pydantic",
        "pyhht",
        "scikit-learn",
        "scipy>=1.8.0",
        "statsmodels",
        "uncertainties",
        "renishawWiRE",  # rc1-parser
        "opusFC",  # rc1-parser
        "spc-spectra",  # rc1-parser
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Libraries',
]


def setup_package():
    setup(
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        classifiers=CLASSIFIERS,
        data_files=DATA_FILES,
        description=DESCRIPTION,
        install_requires=INSTALL_REQUIRES,
        keywords=KEYWORDS,
        license=LICENSE,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        name=NAME,
        package_dir=PACKAGE_DIR,
        packages=PACKAGES,
        python_requires=PYTHON_REQUIRES,
        url=URL,
        version=VERSION,
    )


if __name__ == '__main__':
    setup_package()

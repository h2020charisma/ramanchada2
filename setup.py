import ast
from os import path

from setuptools import find_packages, setup

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

PYTHON_REQUIRES = '>=3.9,<3.14'

PACKAGES = find_packages(where='src')

PACKAGE_DIR = {'': 'src'}

PACKAGE_DATA = {'': ['auxiliary/**/*.txt', 'auxiliary/**/*.dat']}

DATA_FILES = []

INSTALL_REQUIRES = [
        "brukeropusreader==1.*",  # rc1-parser
        "h5py==3.*",
        "lmfit==1.*",
        "matplotlib==3.*",
        "numpy>=1.0,<3.0",
        "pandas==2.*",
        "pydantic~=2.0",
        "emd==0.7.*",
        "renishawWiRE==0.1.*",  # rc1-parser
        "scikit-learn==1.*",
        "scipy>=1.8.0,<2.0",
        "spc-io~=0.2.0",
        "statsmodels==0.14.*",
        "uncertainties==3.*",
        "spe2py~=2.0",
]

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
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

CONSOLE_SCRIPTS = [
        'spg2csv = ramanchada2.standalone.spg2csv:spg2csv',
        'ssl_csv_converter = ramanchada2.standalone.ssl_csv_converter:ssl_csv_converter',
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
        package_data=PACKAGE_DATA,
        include_package_data=True,
        packages=PACKAGES,
        python_requires=PYTHON_REQUIRES,
        url=URL,
        entry_points=dict(console_scripts=CONSOLE_SCRIPTS),
        version=VERSION,
    )


if __name__ == '__main__':
    setup_package()

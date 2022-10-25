from setuptools import setup, find_packages

setup(
    name="ramanchada2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "scipy>=1.8.0",
        "uncertainties",
        "pydantic",
        "lmfit",
        "pandas",
        "sklearn",
    ],
)

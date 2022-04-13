from setuptools import setup, find_packages

setup(
    name="ramanchada2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "h5py",
        "scipy",
        "uncertainties",
    ],
)

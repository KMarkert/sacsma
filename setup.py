import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sacsma",
    version="0.1.0",
    description="Sacramento - Soil Moisture Acccounting model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/kmarkert/sacsma",
    packages=setuptools.find_packages(),
    author="Kel Markert",
    author_email="kel.markert@gmail.com",
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "numba"],
)
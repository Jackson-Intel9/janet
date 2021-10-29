from setuptools import setup, find_packages


PACKAGENAME = "janet"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Janet",
    long_description="Janet",
    install_requires=["numpy", "jax"],
    packages=find_packages(),
    package_data={"janet": ("data/*.txt", "tests/testing_data/*.dat")},
    url="https://github.com/ArgonneCPAC/janet",
)

import numpy
from Cython.Build import cythonize
from setuptools import setup

extensions = cythonize(["microsim/**/*.pyx"], include_path=[numpy.get_include()])
setup(ext_modules=extensions)

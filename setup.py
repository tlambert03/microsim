import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        "microsim/samples/_bresenham.pyx", include_path=[numpy.get_include()]
    ),
)

"""Build the Cython inner-loop extension in-place.
Run:  python sailir/_setup_cython.py build_ext --inplace
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='sailir_cython_inners',
    ext_modules=cythonize(
        ['sailir/_add_sub_inner.pyx',
         'sailir/_enumerate_inner.pyx',
         'sailir/_cic_inner.pyx'],
        language_level=3,
    ),
    zip_safe=False,
)

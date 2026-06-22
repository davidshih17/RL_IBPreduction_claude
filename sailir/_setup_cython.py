"""Build the Cython inner-loop extensions in-place, INSIDE the sailir/ package.

Run:  python sailir/_setup_cython.py build_ext --inplace

Produces sailir/_<name>.cpython-<ver>-<arch>.so (imported as sailir._<name>), so
the compiled kernels live next to the sailir modules that use them (packed_cu,
packed_eq, ibp_env) rather than polluting the repo root. The .so are ABI-specific
and gitignored; rebuild on each cluster / Python version. Each importer falls
back to pure Python when the .so is missing.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize

_KERNELS = ['_add_sub_inner', '_enumerate_inner', '_cic_inner', '_packed_kernels']

setup(
    name='sailir_cython_inners',
    ext_modules=cythonize(
        [Extension(f'sailir.{k}', [f'sailir/{k}.pyx']) for k in _KERNELS],
        language_level=3,
    ),
    zip_safe=False,
)

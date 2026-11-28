"""
setup.py

Optional native build for the solver hot loop.

Transpiles ``solver/_rod_residual.pyx`` (Cython) to a C extension
and builds it in place::

    pip install cython
    python setup.py build_ext --inplace

The compiled module ``solver/_rod_residual.*.so`` takes import
precedence over the pure-Python ``solver/_rod_residual.py``, so the
solver automatically uses the native code afterwards.  Without a
build, the pure-Python module is used; results are bit-identical
either way.  Cython is only needed at build time and is not a
runtime dependency.
"""

from __future__ import annotations

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    ext_modules = []
else:
    ext_modules = cythonize(
        [
            Extension(
                "solver._rod_residual",
                ["solver/_rod_residual.pyx"],
            ),
        ],
        language_level=3,
    )

setup(
    name="trasim",
    version="0.3.0",
    description="Multi-stage transmission simulation",
    ext_modules=ext_modules,
)

from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize('HistogramUtils.pyx', language_level = "3"))
setup(ext_modules = cythonize('JoinGraphUtils.pyx', language_level = "3"))
setup(ext_modules = cythonize('SafeBoundUtils.pyx', language_level = "3"))
setup(ext_modules = cythonize('PiecewiseConstantFunctionUtils.pyx', language_level = "3"))

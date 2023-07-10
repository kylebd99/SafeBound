from setuptools import Extension, setup
from Cython.Build import cythonize

histogram_extension = [
    Extension("HistogramUtils", ["HistogramUtils.pyx"], 
              extra_compile_args=['-fopenmp', "-O3"],
              extra_link_args=['-fopenmp'], language="c++")
]
setup(ext_modules = cythonize(histogram_extension, language_level=3), zip_safe=False)

join_graph_extension = [
    Extension("JoinGraphUtils", ["JoinGraphUtils.pyx"], 
              extra_compile_args=['-fopenmp', "-O3"],
              extra_link_args=['-fopenmp'], language="c++")
]
setup(ext_modules = cythonize(join_graph_extension, language_level=3), zip_safe=False)

safebound_extension = [
    Extension("SafeBoundUtils", ["SafeBoundUtils.pyx"], 
              extra_compile_args=['-fopenmp', "-O3"],
              extra_link_args=['-fopenmp'], language="c++")
]
setup(ext_modules = cythonize(safebound_extension, language_level=3), zip_safe=False)

piecewise_extension = [
    Extension("PiecewiseConstantFunctionUtils", ["PiecewiseConstantFunctionUtils.pyx"], 
              extra_compile_args=['-fopenmp', "-O3"],
              extra_link_args=['-fopenmp'], language="c++")
]
setup(ext_modules = cythonize(piecewise_extension, language_level=3), zip_safe=False)
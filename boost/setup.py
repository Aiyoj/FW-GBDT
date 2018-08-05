# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize(Extension(
    '_gradient_boosting',
    sources=['_gradient_boosting.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
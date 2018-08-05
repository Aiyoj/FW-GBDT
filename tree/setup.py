# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize(
        [
            Extension('_utils', sources=['_utils.pyx'], language='c', include_dirs=[numpy.get_include()]),
            Extension('_splitter', sources=['_splitter.pyx'], language='c', include_dirs=[numpy.get_include()]),
            Extension('_criterion', sources=['_criterion.pyx'], language='c', include_dirs=[numpy.get_include()]),
            Extension('_tree', sources=['_tree.pyx'], language='c', include_dirs=[numpy.get_include()]),
        ]
))


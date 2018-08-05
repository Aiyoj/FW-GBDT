# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import DecisionTreeRegressor
from .tree import DecisionTreeClassifier
from .export import export_graphviz

__all__ = ["DecisionTreeRegressor", "DecisionTreeClassifier", "export_graphviz"]

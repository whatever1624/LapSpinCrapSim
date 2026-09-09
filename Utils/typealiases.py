"""
Collection of aliases to simplify type hinting.

This is primarily to make type hinting NumPy arrays easier.
"""
# Python standard libraries
import typing

# External libraries
import numpy as np

# For ease of copying for imports:
#   from Utils.typeAliases import Any, ListFloat2D, NDArrayFloat1D, NDArrayFloat2D, NDArrayNumber1D

Any = typing.Any

ListFloat2D = list[list[float]]

NDArrayFloat1D = np.ndarray[tuple[int], np.dtype[np.floating]]
NDArrayFloat2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]

NDArrayNumber1D = np.ndarray[tuple[int], np.dtype[np.number]]

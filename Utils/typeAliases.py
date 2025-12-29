"""
Collection of aliases to simplify type hinting.
"""

# Python standard libraries
import typing
import numpy as np

# For ease of copying for imports:
#   from Utils.typeAliases import Any, ListFloat2D, NDArrayFloat1D, NDArrayFloat2D, NDArrayInt1D, NDArrayNumber1D

Any = typing.Any

ListFloat2D = list[list[float]]

NDArrayFloat1D = np.ndarray[tuple[int], np.dtype[np.floating]]
NDArrayFloat2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]

NDArrayInt1D = np.ndarray[tuple[int], np.dtype[np.integer]]

NDArrayNumber1D = np.ndarray[tuple[int], np.dtype[np.number]]

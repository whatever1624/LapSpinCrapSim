"""
Collection of aliases to simplify type hinting.
"""

import numpy as np

ListFloat2D = list[list[float]]

NDArrayFloat1D = np.ndarray[tuple[int], np.dtype[np.floating]]
NDArrayFloat2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]

NDArrayInt1D = np.ndarray[tuple[int], np.dtype[np.integer]]

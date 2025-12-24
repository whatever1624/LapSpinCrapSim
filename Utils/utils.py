"""
Collection of helper functions commonly used by the other modules.
"""

# Import packages
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import shapely

# Import project python files
from typeAliases import *
from track import Track
from trajectory import Trajectory

# Plotting variables
globalOptProgressDict = {'NumEvals': 0, 'EvalResults': [], 'BestResults': []}
globalPlotsDict = {'Fig': plt.figure(), 'TrackTrajDict': {}, 'OptProgressDict': {}, 'LapSimProgressDict': {}}
globalPltPauseDuration = 0.01   # Duration to pause for the matplotlib GUI to update
trackTrajBuffer = 20            # Buffer around the track edges
trackPlotArtists = ['LeftLines', 'RightLines', 'LeftExtendLines', 'RightExtendLines', 'StartLine', 'FinishLine']
trajPlotArtists = ['ControlPoints', 'TrajectoryLines', 'TrackLimitsLinesList']
optProgressPlotArtists = ['ProgressLine', 'BestLine']
lapSimProgressPlotArtists = []


def wrap(x: float | NDArrayFloat1D | NDArrayFloat2D,
         lowerBound: float,
         upperBound: float) -> float | NDArrayFloat1D:
    """
    Wraps value(s) between the lower (inclusive) and upper (exclusive) bounds. Supports NumPy arrays as arguments.

    Args:
        x: Float or NumPy array of floats to wrap.
        lowerBound: Lower bound for the wrapping. This bound is inclusive.
        upperBound: Upper bound for the wrapping. This bound is exclusive.

    Returns:
        If the argument x passed in was a float, returns the wrapped float of x.
        If the argument x passed in was a NumPy array of floats, returns an array of floats in same shape where each element is wrapped.
    """
    return lowerBound + ((x - lowerBound) % (upperBound - lowerBound))


def sideOfLine(xp: float,
               yp: float,
               x1: float,
               y1: float,
               x2: float,
               y2: float) -> float:
    """
    Finds which side of the line the point (xp, yp) is, where the line is from (x1, y1) to (x2, y2) in that direction.

    Args:
        xp: x coordinate of the point.
        yp: y coordinate of the point.
        x1: x coordinate of the start of the line.
        y1: y coordinate of the start of the line.
        x2: x coordinate of the end of the line.
        y2: y coordinate of the end of the line.

    Returns:
        0 if all points are collinear (i.e. all points are on the same straight line).
        >0 if the point (xp, yp) is on the right of the line.
        <0 if the point (xp, yp) is on the left of the line.
    """
    return ((xp - x1) * (y2 - y1)) - ((yp - y1) * (x2 - x1))


def rotateVector2D(xyVector: NDArrayFloat1D,
                   theta: float) -> NDArrayFloat1D:
    """
    Rotates the 2D vector anti-clockwise by theta radians.

    Args:
        xyVector: NumPy array in the form [x, y] representing the 2D vector.
        theta: Angle in radians to rotate the vector, in the anti-clockwise direction.

    Returns:
        NumPy array in the form [x*, y*] representing the rotated vector.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    xyVectorRotated = np.empty(2)
    xyVectorRotated[0] = (c * xyVector[0]) - (s * xyVector[1])
    xyVectorRotated[1] = (s * xyVector[0]) + (c * xyVector[1])
    return xyVectorRotated

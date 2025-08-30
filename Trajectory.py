"""Trajectory class and its related functions"""

# Import packages
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import shapely

# Import LapSim project python files
import utils
from Track import Track


class Trajectory:
    def __init__(self, trajType: str, track: Track, CP: list[list[float]] | np.ndarray[tuple[float, float], np.dtype[float]], sDelta: float, degree: int=3) -> None:
        # Check that trajectory type is valid and finishGate is passed if required
        allowedTypes = ['Closed Circuit', 'Point to Point', 'Point to Point with Run Up', 'Single Lap']
        if trajType in allowedTypes:
            self.trajType = trajType
        else:
            raise Exception("\'" + trajType + "\' is not a valid trajectory type. Valid trajectory types are " + str(allowedTypes))

        # Convert control points coordinate list/array to NumPy array
        CP = np.array(CP)
        self.CP = CP.copy

        # If the trajectory type is 'Closed Circuit', make the trajectory spline periodic and closed
        if trajType == 'Closed Circuit':
            bc_type = 'periodic'
            if not np.array_equal(CP[0], CP[-1]):
                CP = np.vstack((CP, CP[0]))
        else:
            bc_type = None

        # Create trajectory spline object
        nCP = np.arange(len(CP))
        self.spline = scipy.interpolate.make_interp_spline(nCP, CP, k=degree, bc_type=bc_type)

        # Calculate the ds/dp array (rate of change of distance with control point value) - idk if this is compatible https://stackoverflow.com/a/2184220
        # and find the pStart and pFinish (control point values at the start and finish gate - these will be done by start/finish gate intersections
        # and if there are multiple intersections then choose the one that is closest to the gateMidpoint and with the direction dot product closest to 1)
        # NOTE: this will need to consider the track z height so that distance is always longitudinal to the trajectory even for steep up/downhill
        #       possibly make another array that's dz/ds at 0.1m intervals of s or something idk then interpolate on that for the actual trajectory z

        # Handling for the trajectory spline not intersecting the start or finish gates

        # Calculate cumulative distance along the pSamples array by integrating ds/dp with respect to p

        # Calculate total distance from start gate to finish gate, offset cumulative distance array so it's 0 at the start line
        # and wrap distances if the trajectory is 'Closed Circuit'

        # Discretize the trajectory spline with discretization step as close to sDelta as possible

        # Calculate trajectory slope, dSlope_ds, camber, curvature (probably helps if i calculate trajectory direction vector and normal vector first)

        # Track limits

        pass # Trajectory initialised :)

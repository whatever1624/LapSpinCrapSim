"""
Main place to configure and run the lap sim.

However, currently just being used as a sandbox to test the lap sim modules.
"""

# Import packages
import time
import scipy
import shapely
import numpy as np
import matplotlib.pyplot as plt

# Import project python files
from Utils.typeAliases import *
from Utils import utils
from track import Track

"""left = [[1, 2, 3], [0, 2, 4]]
right = [[3, 4, 5], [1, 1, 5.5], [1, 2, 6]]
leftEx = [[-0.2, 0, 1], [-0.1, 2, 2]]
rightEx = [[1.1, 0, 7], [1.15, 1, 7.5], [1.2, 2, 8]]"""


def generateTrackLimits(straightLength: float,
                        cornerRadius: float,
                        trackWidth: float) -> tuple[NDArrayFloat2D, NDArrayFloat2D]:
    """Anti-clockwise oval with 2 corners"""
    straightStep = 5
    cornerPoints = 15

    halfStraightLength = straightLength / 2
    halfTrackWidth = trackWidth / 2

    # Starting half straight
    startHalfStraight = np.arange(0, halfStraightLength, straightStep).tolist()
    xLeft = startHalfStraight.copy()
    yLeft = np.full_like(xLeft, trackWidth).tolist()
    xRight = startHalfStraight.copy()
    yRight = np.zeros_like(xRight).tolist()

    # Turn 1
    thetas = np.linspace(np.pi, 0, cornerPoints)
    xLeft += (np.sin(thetas) * (cornerRadius - halfTrackWidth) + halfStraightLength).tolist()
    yLeft += (np.cos(thetas) * (cornerRadius - halfTrackWidth) + cornerRadius + halfTrackWidth).tolist()
    xRight += (np.sin(thetas) * (cornerRadius + halfTrackWidth) + (straightLength / 2)).tolist()
    yRight += (np.cos(thetas) * (cornerRadius + halfTrackWidth) + cornerRadius + halfTrackWidth).tolist()

    # Back straight
    fullStraight = np.arange(halfStraightLength - straightStep, -halfStraightLength, -straightStep).tolist()
    xLeft += fullStraight.copy()
    yLeft += np.full_like(fullStraight, 2 * cornerRadius).tolist()
    xRight += fullStraight.copy()
    yRight += np.full_like(fullStraight, 2 * cornerRadius + trackWidth).tolist()

    # Turn 2
    thetas = np.linspace(0, -np.pi, cornerPoints)
    xLeft += (np.sin(thetas) * (cornerRadius - halfTrackWidth) - halfStraightLength).tolist()
    yLeft += (np.cos(thetas) * (cornerRadius - halfTrackWidth) + cornerRadius + halfTrackWidth).tolist()
    xRight += (np.sin(thetas) * (cornerRadius + halfTrackWidth) - (straightLength / 2)).tolist()
    yRight += (np.cos(thetas) * (cornerRadius + halfTrackWidth) + cornerRadius + halfTrackWidth).tolist()

    # Finishing half straight
    finishHalfStraight = np.arange(-halfStraightLength + straightStep, 0, straightStep).tolist()
    xLeft += finishHalfStraight.copy()
    yLeft += np.full_like(finishHalfStraight, trackWidth).tolist()
    xRight += finishHalfStraight.copy()
    yRight += np.zeros_like(finishHalfStraight).tolist()

    # Convert lists to NumPy arrays
    xLeft = np.array(xLeft)
    xRight = np.array(xRight)
    yLeft = np.array(yLeft)
    yRight = np.array(yRight)

    # Create left and right z arrays - currently just all 0
    zLeft = np.zeros_like(xLeft)
    zRight = np.zeros_like(xRight)

    # Combine separate arrays into coordinate arrays
    left = np.vstack((xLeft, yLeft, zLeft)).T
    right = np.vstack((xRight, yRight, zRight)).T

    return left, right


left, right = generateTrackLimits(200, 50, 20)
leftExtend, rightExtend = generateTrackLimits(195, 49.5, 21)

track = Track(left, right, leftExtend, gateStep=20)
#track = Track(left, right, leftExtend, rightExtend)
#track = Track(leftExtend, rightExtend)

utils.updateTrack(track)
utils.refreshPlot()

#controlPoints = [[90, 10], [150, 60], [90, 110], [-90, 110], [-150, 60], [-90, 10]]
#traj = Trajectory('Closed Circuit', track, controlPoints, 1)

plt.show()

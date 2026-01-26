"""
Script to generate a simple oval track for testing.

The generated track:
    -   Has banking in T1 between the soft track limits, but flat track in T2.
    -   Has a very lenient hard track limit on the inside of T2.
    -   Option to provide extra coordinate arrays.
    -   Option to add noise to the generated coordinates.
    -   Option to generate as a closed or open/point-to-point track - with the
        open track starting at the start of T1 and finishing at the end of T2.
    -   TODO: Option for custom start and finish gates (with the same locations
              as the start and end of the point-to-point track).
"""
# Python standard libraries
import os

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from Utils.typeAliases import NDArrayFloat2D
from Utils import utils

# Oval track parameters
saveFolder = r"OvalTestTrack"   # Folder to save the track data files to
BClosed = True                  # Whether the track should be closed or open - open being omitting the normal start/finish straight
AHeadingStart = -2              # Heading angle on the start/finish straight (radians)
ABankingMax = np.pi / 4         # Maximum banking angle in the middle of T1 (radians)
lStep = 1                       # Approximate distance between coordinates (m)
lStraight = 200                 # Length of the start/finish or back straight (m)
lRadius = 50                    # Corner radius of the centreline of the track (m)
lWidth = 15                     # Width of the track (m)
lExtend = 1                     # Additional width of the track on each side to the hard track limits (m)
lNoise = 0.2                    # Standard deviation of noise in all the x, y, z coordinates (m)
nExtraCoordArrays = 3           # Number of extra coordinate arrays


def getOvalCoords(lRadius: float,
                  zMax: float,
                  xOffset: float,
                  yOffset: float,
                  BSquareT2: bool) -> NDArrayFloat2D:
    xyzCoords = np.empty((0, 3))

    # Starting half straight
    if BClosed:
        n = int(np.floor(((lStraight / 2) - lStep) / lStep))
        xyz = np.zeros((n, 3))
        xyz[:, 1] = np.linspace(0, (lStraight / 2) - lStep, n)
        xyzCoords = np.vstack((xyzCoords, xyz))

    # Turn 1
    n = int(np.floor(np.pi * lRadius / lStep))
    theta = np.linspace(np.pi / 2, -np.pi / 2, n)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = (np.sin(theta) * lRadius) - lRadius
    xyz[:, 1] = (np.cos(theta) * lRadius) + (lStraight / 2)
    xyz[:, 2] = np.square(np.cos(theta)) * zMax
    xyzCoords = np.vstack((xyzCoords, xyz))

    # Back straight
    n = int(np.floor((lStraight - (2 * lStep)) / lStep))
    xyz = np.zeros((n, 3))
    xyz[:, 0] -= 2 * lRadius
    xyz[:, 1] = np.linspace((lStraight / 2) - lStep, -(lStraight / 2) + lStep, n)
    xyzCoords = np.vstack((xyzCoords, xyz))

    # Turn 2
    if BSquareT2:
        n = int(np.floor(2 * lRadius / lStep))
        xyz = np.zeros((n, 3))
        xyz[:, 0] = np.linspace(-lRadius, lRadius, n) - lRadius
        xyz[:, 1] -= lStraight / 2
    else:
        n = int(np.floor(np.pi * lRadius / lStep))
        theta = np.linspace(-np.pi / 2, -3 * np.pi / 2, n)
        xyz = np.zeros((n, 3))
        xyz[:, 0] = (np.sin(theta) * lRadius) - lRadius
        xyz[:, 1] = (np.cos(theta) * lRadius) - (lStraight / 2)
    xyzCoords = np.vstack((xyzCoords, xyz))

    # Finishing half-straight
    if BClosed:
        n = int(np.floor((lStraight / 2) - (2 * lStep) / lStep))
        xyz = np.zeros((n, 3))
        xyz[:, 1] = np.linspace(-(lStraight / 2) + lStep, -lStep, n)
        xyzCoords = np.vstack((xyzCoords, xyz))

    # Offsets, rotations and add noise
    xyzCoords[:, 0] += xOffset
    xyzCoords[:, 1] += yOffset
    for i, xyz in enumerate(xyzCoords):
        xyzCoords[i] = utils.rotateVectorHeading(xyz, AHeadingStart)
    xyzCoords += np.random.normal(0, lNoise, np.shape(xyzCoords))

    return xyzCoords


# Generate track limits
xyzCoordsDict = {'xyzLimitLeftHard': getOvalCoords(lRadius - (lWidth / 2) - lExtend, 0, -(lWidth / 2) - lExtend, 0, True),
                 'xyzLimitLeftSoft': getOvalCoords(lRadius - (lWidth / 2), 0, -(lWidth / 2), 0, False),
                 'xyzLimitRightSoft': getOvalCoords(lRadius + (lWidth / 2), lRadius * np.sin(ABankingMax), lWidth / 2, 0, False),
                 'xyzLimitRightHard': getOvalCoords(lRadius + (lWidth / 2) + lExtend, (lRadius + lExtend) * np.sin(ABankingMax), lWidth / 2 + lExtend, 0, False)}

# Generate additional coordinate arrays
if nExtraCoordArrays > 0:
    lR = np.linspace(lRadius - (lWidth / 2), lRadius + (lWidth / 2), nExtraCoordArrays + 2)
    z = np.linspace(0, lRadius * np.sin(ABankingMax), nExtraCoordArrays + 2)
    x = np.linspace(-(lWidth / 2), lWidth / 2, nExtraCoordArrays + 2)
    y = np.linspace(0, 0, nExtraCoordArrays + 2)
    for i in range(1, nExtraCoordArrays + 1):
        xyzCoordsDict[f'xyzExtra{i}'] = getOvalCoords(lR[i], z[i], x[i], y[i], False)

# Plot result and save to CSV
if not os.path.isdir(saveFolder):
    os.makedirs(saveFolder)
zMin = min([min(xyz[:, 2]) for xyz in xyzCoordsDict.values()])
zMax = max([max(xyz[:, 2]) for xyz in xyzCoordsDict.values()])
for key, xyzCoords in xyzCoordsDict.items():
    plt.scatter(xyzCoords[:, 0], xyzCoords[:, 1], c=xyzCoords[:, 2], cmap='magma', marker='.', vmin=zMin, vmax=zMax, label=key)
    np.savetxt(os.path.join(saveFolder, key + ".csv"), xyzCoords, fmt='%1.18f', delimiter=',')
plt.axis('equal')
plt.legend()
plt.show()

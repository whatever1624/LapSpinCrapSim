"""
Main place to configure and run the lap sim.

However, currently just being used as a sandbox to test the lap sim modules.
"""
# Python standard libraries
...

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from Utils import utils
from track import Track

trackTest = Track(r"C:\Users\Willow\Documents\Repos\LapSpinCrapSim\Tracks\BrandsHatchIndy", True, BDebug=False)

# Plot track z coordinate, slope and camber at the gate midpoint along the gates
fig, axs = plt.subplots(2, 1, layout='constrained')
zMidList = []
zLeftSoftList = []
zRightSoftList = []
ASlopeList = []
ACamberList = []
for i, gate in enumerate(trackTest.gates):
    zMid = trackTest.getTrackZ(gate.xyMidpoint, i)
    zLeftSoft = trackTest.getTrackZ(gate.xyMidpoint + (np.array(gate.xyLine.xy)[:, 0] - gate.xyMidpoint) / gate.lLeft * gate.lLimitLeftSoft, i)
    zRightSoft = trackTest.getTrackZ(gate.xyMidpoint + (np.array(gate.xyLine.xy)[:, 1] - gate.xyMidpoint) / gate.lRight * gate.lLimitRightSoft, i)
    xyzNormal = trackTest.getTrackNormal(gate.xyMidpoint, i, 6)

    gateStr = f"Gate {i} [{gate.xyMidpoint[0]:.1f}, {gate.xyMidpoint[1]:.1f}]"
    print(f"{gateStr}{" " * (30 - len(gateStr))}{zMid:.3f}\t{xyzNormal}")

    zMidList.append(zMid)
    zLeftSoftList.append(zLeftSoft)
    zRightSoftList.append(zRightSoft)

    ASlopeList.append((np.acos(np.dot(xyzNormal, utils.rotateVectorHeading([0, 1, 0], gate.AHeading))) - np.pi / 2) * 180 / np.pi)
    ACamberList.append((np.acos(np.dot(xyzNormal, utils.rotateVectorHeading([-1, 0, 0], gate.AHeading))) - np.pi / 2) * 180 / np.pi)

axs[0].plot(zMidList, marker='.', ls='-', label='Mid')
axs[0].plot(zLeftSoftList, marker='.', ls='-', label='Left Soft')
axs[0].plot(zRightSoftList, marker='.', ls='-', label='Right Soft')

axs[1].plot(ASlopeList, ls='-', label='Slope')
axs[1].plot(ACamberList, ls='-', label='Camber')

axs[0].legend()
axs[1].legend()
axs[0].set_ylabel("Track z coordinate (m)")
axs[1].set_ylabel("Track slope and camber (deg)")
axs[1].set_ylim([-10, 10])
axs[1].set_xlabel("Gate index")
plt.show()

# Plot track z coordinate along the width of each gate
lResolution = 0.01
xPlotRange = 30
yPlotRange = 30
fig, ax = plt.subplots(1, 1, layout='constrained')
for i, gate in enumerate(trackTest.gates):
    ax.clear()

    # Plot gate intersections with track limits
    keys = ['LimitLeftSoft', 'LimitRightSoft', 'LimitLeftHard', 'LimitRightHard']
    markerList = ['<', '>', '<', '>']
    fillList = ['none', 'none', 'full', 'full']
    cDict = {'LimitLeftSoft': (1, 0, 0), 'LimitRightSoft': (0, 1, 0), 'LimitLeftHard': (0.5, 0, 0), 'LimitRightHard': (0, 0.5, 0)}
    xyVec = utils.rotateVectorHeading(np.array([1, 0]), gate.AHeading)
    lLimits = [-gate.lLimitLeftSoft, gate.lLimitRightSoft, -gate.lLimitLeftHard, gate.lLimitRightHard]
    for j, key in enumerate(keys):
        z = trackTest.getTrackZ(gate.xyMidpoint + (xyVec * lLimits[j]), i)
        ax.plot(lLimits[j], z, c=cDict[key], fillstyle=fillList[j], ls='', marker=markerList[j])

    lList = np.arange(np.floor(-gate.lLimitLeftHard / lResolution) * lResolution,
                      np.ceil(gate.lLimitRightHard / lResolution) * lResolution + lResolution,
                      lResolution)
    indlZero = np.argmin(np.abs(lList))
    zList = []
    for l in lList:
        zList.append(trackTest.getTrackZ(gate.xyMidpoint + (xyVec * l), i))
    ax.plot(lList, zList)
    ax.set_xlim((-xPlotRange / 2, xPlotRange / 2))
    ax.set_ylim((zList[indlZero] - yPlotRange, zList[indlZero] + yPlotRange))
    plt.pause(0.01)

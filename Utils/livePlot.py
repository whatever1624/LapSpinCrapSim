"""
Module for live plotting of the lap sim progress and optimisation progress.
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


def getAxsIndex(axsName: str) -> int:
    """
    Returns the index for the axs corresponding to the axes name.

    Args:
        axsName: One of the strings 'TrackTraj', 'OptProgress', 'LapSimProgress' representing the name of the axes.

    Returns:
        Index for the axs corresponding to the axes name.

    Raises:
        Exception: 'axsName' is not a value axes name. Valid axes names are ['TrackTraj', 'OptProgress', 'LapSimProgress'].
    """
    if axsName == 'TrackTraj':
        return 0
    elif axsName == 'OptProgress':
        return 1 if globalPlotsDict['TrackTrajDict'] else 0
    elif axsName == 'LapSimProgress':
        return 2 if globalPlotsDict['TrackTrajDict'] and globalPlotsDict['OptProgressDict'] else 1 if globalPlotsDict['TrackTrajDict'] or globalPlotsDict['OptProgressDict'] else 0
    else:
        raise Exception("\'" + axsName + "\' is not a valid axes name. Valid axes names are " + str(['TrackTraj', 'OptProgress', 'LapSimProgress']) + ".")


def updateTrack(track: Track) -> None:
    """
    Updates the Track object stored in globalPlotsDict to the new Track object passed in.

    Args:
        track: New Track object to replace the old Track object stored in globalPlotsDict.
    """
    globalPlotsDict['TrackTrajDict']['Track'] = track


def refreshPlot() -> None:
    """
    Clears plot then re-plots everything based on the data in globalPlotsDict.
    """
    global globalPlotsDict

    trackTrajDict = globalPlotsDict['TrackTrajDict']
    optProgressDict = globalPlotsDict['OptProgressDict']
    lapSimProgressDict = globalPlotsDict['LapSimProgressDict']

    # Clear plot
    globalPlotsDict['Fig'].clear()

    # Remove plot artist entries from the plot dictionaries
    for key in list(trackTrajDict.keys()):
        if key in trackPlotArtists or key in trajPlotArtists:
            del trackTrajDict[key]
    for key in list(optProgressDict.keys()):
        if key in optProgressPlotArtists:
            del optProgressDict[key]
    for key in list(lapSimProgressDict.keys()):
        if key in lapSimProgressPlotArtists:
            del lapSimProgressDict[key]

    # If the track trajectory is not empty
    if trackTrajDict:
        # Add the subplot and plot from the stored Track and Trajectory objects in the plot dictionary
        globalPlotsDict['Fig'].add_subplot()
        plotTrack()
        plotTraj()

    # if the optimisation progress is not empty
    if optProgressDict:
        # Add the subplot and plot from the globalOptProgressDict data
        globalPlotsDict['Fig'].add_subplot()
        plotOptProgress()

    # if the lap sim progress is not empty
    if lapSimProgressDict:
        # Add the subplot and idk i've not even made the lap sim so who knows what's needed here
        globalPlotsDict['Fig'].add_subplot()
        plotLapSimProgress()


def plotTrack() -> None:
    """
    Removes any old track-related plot artists, plots the track in the live plot based on the Track object stored in globalPlotsDict,
    then updates the plot dictionary with the new plot artists.
    """
    global globalPlotsDict

    ax = globalPlotsDict['Fig'].get_axes()[getAxsIndex('TrackTraj')]
    trackTrajDict = globalPlotsDict['TrackTrajDict']
    track: Track | None = trackTrajDict.get('Track', None)  # Type hint just here so PyCharm knows "if track" condition can be true

    # Remove old plot artists
    for key in list(trackTrajDict.keys()):
        if key in trackPlotArtists:
            trackTrajDict[key].pop().remove()

    if track:
        # Set axes limits and make the axes square
        ax.axis('square')
        ax.set_xlim(track.xMin - trackTrajBuffer, track.xMax + trackTrajBuffer)
        ax.set_ylim(track.yMin - trackTrajBuffer, track.yMax + trackTrajBuffer)

        # Plot LeftExtendLines and RightExtendLines (plot these first so LeftLines and RightLines plot over these)
        leftExtend = track.gatesMidpoint + np.transpose([-track.gatesDirection[:, 1] * track.leftExtendWidths, track.gatesDirection[:, 0] * track.leftExtendWidths])
        rightExtend = track.gatesMidpoint + np.transpose([track.gatesDirection[:, 1] * track.rightExtendWidths, -track.gatesDirection[:, 0] * track.rightExtendWidths])
        if track.isClosed:
            leftExtend = np.vstack((leftExtend, leftExtend[0]))
            rightExtend = np.vstack((rightExtend, rightExtend[0]))
        trackTrajDict['LeftExtendLines'] = ax.plot(leftExtend[:, 0], leftExtend[:, 1], c='grey')
        trackTrajDict['RightExtendLines'] = ax.plot(rightExtend[:, 0], rightExtend[:, 1], c='grey')

        # Plot LeftLines and RightLines
        left = track.gatesMidpoint + np.transpose([-track.gatesDirection[:, 1] * track.leftWidths, track.gatesDirection[:, 0] * track.leftWidths])
        right = track.gatesMidpoint + np.transpose([track.gatesDirection[:, 1] * track.rightWidths, -track.gatesDirection[:, 0] * track.rightWidths])
        if track.isClosed:
            left = np.vstack((left, left[0]))
            right = np.vstack((right, right[0]))
        trackTrajDict['LeftLines'] = ax.plot(left[:, 0], left[:, 1], c='k')
        trackTrajDict['RightLines'] = ax.plot(right[:, 0], right[:, 1], c='k')

        # DEBUG - Plot gates (both from left/right and leftExtend/rightExtend)
        for i in range(np.size(track.gates)):
            ax.plot([leftExtend[i, 0], rightExtend[i, 0]], [leftExtend[i, 1], rightExtend[i, 1]], c='C3')
            ax.plot([left[i, 0], right[i, 0]], [left[i, 1], right[i, 1]], c='C2')

        # DEBUG - Plot gates at their full width
        """for gate in track.gates:
            xy = gate.xy
            ax.plot(xy[0], xy[1])"""

        # Plot FinishLine - (plot this first so StartLine plots over this)
        fgi = track.finishGateIndex
        trackTrajDict['FinishLine'] = ax.plot([left[fgi, 0], right[fgi, 0]], [left[fgi, 1], right[fgi, 1]])

        # Plot StartLine
        sgi = track.startGateIndex
        trackTrajDict['StartLine'] = ax.plot([left[sgi, 0], right[sgi, 0]], [left[sgi, 1], right[sgi, 1]])

        # Let the plot update
        plt.pause(globalPltPauseDuration)
    else:
        print("No Track object in globalPlotsDict['TrackTrajDict']")


def plotTraj() -> None:
    """
    Removes any old track-related plot artists and plots the trajectory in the live plot, then updates the plot dictionary with the new artists plotted.
    """
    axsIndex = getAxsIndex('TrackTraj')
    trackTrajDict = globalPlotsDict['TrackTrajDict']
    trajectory: Trajectory | None = trackTrajDict.get('Trajectory', None)  # Type hint just here so PyCharm knows "if track" condition can be true

    # Remove old plot artists
    for key in list(trackTrajDict.keys()):
        if key in trajPlotArtists:
            trackTrajDict[key].pop().remove()

    if trajectory:
        ...
        print("IDK WHAT THE TRAJECTORY PLOTTING IS")
    else:
        print("No Trajectory object in globalPlotsDict['TrackTrajDict']")


def plotOptProgress() -> None:
    """
    Removes any old optimisation progress-related plot artists and plots the optimisation progress in the live plot,
    then updates the plot dictionary with the new artists plotted.
    """
    axsIndex = getAxsIndex('OptProgress')
    optProgressDict = globalPlotsDict['OptProgressDict']

    # Remove old plot artists
    for key in list(optProgressDict.keys()):
        if key in optProgressPlotArtists:
            optProgressDict[key].pop().remove()


def plotLapSimProgress() -> None:
    """
    Removes any old lap sim progress-related plot artists and plots the lap sim progress in the live plot
    idk what this will be or if it'll even be needed i've not even made the lap sim
    MAKE SURE TO CHECK REFRESHPLOT() AND MAKE SURE IT DOESN'T CLEAR ANY LAPSIM RESULTS OBJECT OR SMTH IF THAT'S IMPORTANT
    """
    axsIndex = getAxsIndex('LapSimProgress')
    lapSimProgressDict = globalPlotsDict['LapSimProgressDict']

    # Remove old plot artists
    for key in list(lapSimProgressDict.keys()):
        if key in lapSimProgressDict:
            lapSimProgressDict[key].pop().remove()

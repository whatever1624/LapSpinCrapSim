"""
Module for live plotting of the lap sim progress and optimisation progress.
"""

# Import packages
import time
import scipy
import shapely
import numpy as np
import matplotlib.pyplot as plt

# Import project python files
from typeAliases import *
from track import Track
from trajectory import Trajectory

# Constants
TRACK_PLOT_ARTISTS = ['LeftLines', 'RightLines', 'LeftExtendLines', 'RightExtendLines', 'StartLine', 'FinishLine']
TRAJ_PLOT_ARTISTS = ['ControlPoints', 'TrajectoryLines', 'TrackLimitsLinesList']
OPT_PROG_PLOT_ARTISTS = ['ProgressLine', 'BestLine']
LAP_SIM_PLOT_ARTISTS = []

# Global variables
PLOTS_DICT = {'Fig': plt.figure(),
              'TrackTrajDict': {},
              'OptProgressDict': {},
              'LapSimProgressDict': {}}
PLT_PAUSE_DURATION = 0.01   # Duration to pause for the matplotlib GUI to update
TRACK_BUFFER = 20            # Buffer around the track edges


def getAxsIndex(axsName: str) -> int:
    """
    Returns the index for the axs corresponding to the axes name.

    Args:
        axsName: One of the strings 'TrackTraj', 'OptProgress', 'LapSimProgress'
            representing the name of the axes.

    Returns:
        Index for the axs corresponding to the axes name.

    Raises:
        Exception: 'AXSNAME' is not a valid axes name. Valid axes names are
            ['TrackTraj', 'OptProgress', 'LapSimProgress'].
    """
    if axsName == 'TrackTraj':
        return 0
    elif axsName == 'OptProgress':
        return 1 if PLOTS_DICT['TrackTrajDict'] else 0
    elif axsName == 'LapSimProgress':
        return 2 if PLOTS_DICT['TrackTrajDict'] and PLOTS_DICT['OptProgressDict'] else 1 if PLOTS_DICT['TrackTrajDict'] or PLOTS_DICT['OptProgressDict'] else 0
    else:
        raise Exception("\'" + axsName + "\' is not a valid axes name. Valid axes names are " + str(['TrackTraj', 'OptProgress', 'LapSimProgress']) + ".")


def updateTrack(track: Track) -> None:
    """
    Updates the Track object stored in PLOTS_DICT to the new Track object.

    Args:
        track: New Track object to replace the old Track object stored in
            PLOTS_DICT.
    """
    PLOTS_DICT['TrackTrajDict']['Track'] = track


def removePlotArtists(plotArtistsDict: dict[str, Any],
                      plotArtistsList: list[str]) -> None:
    """
    Removes plot artists from plotArtistsDict (both from the dictionary and from
    the live plot), as specified by plotArtistsList.

    Args:
        plotArtistsDict: Dictionary containing the plot artists to be removed.
        plotArtistsList: List of plot artists that should be removed. Only plot
            artists with keys exactly matching elements in this list will be
            removed from plotArtistsDict.
    """
    for key in list(plotArtistsDict.keys()):
        if key in plotArtistsList:
            plotArtist = plotArtistsDict[key].pop().remove()
            try:
                plotArtist.remove()
            except KeyError:
                pass    # Plot artist already doesn't exist


def refreshPlot() -> None:
    """
    Clears plot then re-plots everything based on the data in PLOTS_DICT.
    """
    global PLOTS_DICT

    trackTrajDict = PLOTS_DICT['TrackTrajDict']
    optProgressDict = PLOTS_DICT['OptProgressDict']
    lapSimProgressDict = PLOTS_DICT['LapSimProgressDict']

    # Clear plot
    PLOTS_DICT['Fig'].clear()

    # Remove plot artist entries from the plot dictionaries
    removePlotArtists(trackTrajDict, TRACK_PLOT_ARTISTS)
    removePlotArtists(trackTrajDict, TRAJ_PLOT_ARTISTS)
    removePlotArtists(optProgressDict, OPT_PROG_PLOT_ARTISTS)
    removePlotArtists(lapSimProgressDict, LAP_SIM_PLOT_ARTISTS)

    # If the track trajectory is not empty
    if trackTrajDict:
        # Add the subplot and plot from the stored Track and Trajectory objects in the plot dictionary
        PLOTS_DICT['Fig'].add_subplot()
        plotTrack()
        plotTraj()

    # if the optimisation progress is not empty
    if optProgressDict:
        # Add the subplot and plot from the globalOptProgressDict data
        PLOTS_DICT['Fig'].add_subplot()
        plotOptProgress()

    # if the lap sim progress is not empty
    if lapSimProgressDict:
        # Add the subplot and idk i've not even made the lap sim so who knows what's needed here
        PLOTS_DICT['Fig'].add_subplot()
        plotLapSimProgress()


def plotTrack() -> None:
    """
    Updates the track on the live plot.

    First removes any old track-related plot artists, then plots the track in
    the live plot based on the Track object stored in PLOTS_DICT, also updating
    PLOTS_DICT with the new plot artists.
    """
    global PLOTS_DICT

    ax = PLOTS_DICT['Fig'].get_axes()[getAxsIndex('TrackTraj')]
    trackTrajDict = PLOTS_DICT['TrackTrajDict']
    track: Track | None = trackTrajDict.get('Track', None)  # Type hint just here so PyCharm knows "if track" condition can be true

    removePlotArtists(trackTrajDict, TRACK_PLOT_ARTISTS)

    if track:
        # Set axes limits and make the axes square
        ax.axis('square')
        ax.set_xlim(track.xMin - TRACK_BUFFER, track.xMax + TRACK_BUFFER)
        ax.set_ylim(track.yMin - TRACK_BUFFER, track.yMax + TRACK_BUFFER)

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
        plt.pause(PLT_PAUSE_DURATION)
    else:
        print("No Track object in globalPlotsDict['TrackTrajDict']")


def plotTraj() -> None:
    """
    Updates the trajectory on the live plot.

    First removes any old trajectory-related plot artists, then plots the
    trajectory in the live plot based on the Trajectory object stored in
    PLOTS_DICT, also updating PLOTS_DICT with the new plot artists.

    TODO: Finish this - the plotting and updating PLOTS_DICT with the new plot
        artists have not been implemented yet.
    """
    axsIndex = getAxsIndex('TrackTraj')
    trackTrajDict = PLOTS_DICT['TrackTrajDict']
    trajectory: Trajectory | None = trackTrajDict.get('Trajectory', None)  # Type hint just here so PyCharm knows "if track" condition can be true

    removePlotArtists(trackTrajDict, TRAJ_PLOT_ARTISTS)

    if trajectory:
        ...
        print("IDK WHAT THE TRAJECTORY PLOTTING IS")
    else:
        print("No Trajectory object in globalPlotsDict['TrackTrajDict']")


def plotOptProgress() -> None:
    """
    Updates the optimisation progress on the live plot.

    First removes any old optimisation progress-related plot artists, then plots
    the optimisation progress in the live plot based on OPT_PROGRESS_DICT, also
    updating PLOTS_DICT with the new plot artists.

    TODO: Finish this - the plotting and updating PLOTS_DICT with the new plot
        artists have not been implemented yet.
    """
    axsIndex = getAxsIndex('OptProgress')
    optProgressDict = PLOTS_DICT['OptProgressDict']

    removePlotArtists(optProgressDict, OPT_PROG_PLOT_ARTISTS)


def plotLapSimProgress() -> None:
    """
    Updates the lap sim progress on the live plot.

    First removes any old lap sim-related plot artists, then plots the lap sim
    progress in the live plot based on the LapSimProgressDict dictionary stored
    in PLOTS_DICT, also updating PLOTS_DICT with the new plot artists.

    TODO: Finish this - the plotting and updating PLOTS_DICT with the new plot
        artists have not been implemented yet.
    """
    axsIndex = getAxsIndex('LapSimProgress')
    lapSimProgressDict = PLOTS_DICT['LapSimProgressDict']

    removePlotArtists(lapSimProgressDict, LAP_SIM_PLOT_ARTISTS)

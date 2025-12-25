"""
Track Class and its related functions.

The Track object defines the track on which a trajectory can be created and
optimised. More documentation about the Track object will come when I get around
to making the Track object.
"""

# Import packages
import time
import scipy
import shapely
import numpy as np
import matplotlib.pyplot as plt

# Import project python files
from Utils import utils
from Utils.typeAliases import *


def getLimitsDistances(limits: NDArrayFloat2D) -> NDArrayFloat1D:
    """
    Calculates the array of the cumulative distance along the limits
    coordinates, assuming straight lines between limits coordinates.

    Args:
        limits: 2D array where each element is an [x, y, z] coordinate of the
            track limit, and each element increases the distance along the
            track.

    Returns:
        1D array representing the cumulative distance along the limits
        coordinates, assuming straight lines between limits coordinates.
    """
    n = np.size(limits, 0)
    distances = np.empty(n)
    distances[0] = 0
    for i in range(1, n):
        distances[i] = distances[i - 1] + scipy.linalg.norm(limits[i] - limits[i - 1])

    return distances


def getGateFromCoords(leftCoord: NDArrayFloat1D,
                      rightCoord: NDArrayFloat1D,
                      gateHalfWidth: float) -> tuple[shapely.LineString, NDArrayFloat1D, NDArrayFloat1D]:
    """
    Calculates the gate passing through the input coordinates. Only the x and y
    coordinates are used for the gate calculation.

    Args:
        leftCoord: 1D array of the left coordinate of the gate, in the form
            [x, y] or [x, y, z].
        rightCoord: 1D array of the right coordinate of the gate, in the form
            [x, y] or [x, y, z].
        gateHalfWidth: Half-width of the gate.

    Returns:
        Tuple of (gate, gateMidpoint, gateDirection).

        gate: Shapely LineString representing the gate.

        gateMidpoint: Coordinates of the midpoint of the gate, in the form
        [x, y].

        gateDirection: Direction vector of the gate in the direction of forward
        travel, normalised to a magnitude of 1.
    """
    gateMidpoint = np.array([(leftCoord[0] + rightCoord[0]) / 2, (leftCoord[1] + rightCoord[1]) / 2])
    dx = rightCoord[0] - leftCoord[0]
    dy = rightCoord[1] - leftCoord[1]
    norm = scipy.linalg.norm([dx, dy])
    gateDirection = np.array([-dy / norm, dx / norm])
    gateLeft = gateMidpoint + ([-gateDirection[1] * gateHalfWidth, gateDirection[0] * gateHalfWidth])
    gateRight = gateMidpoint + ([gateDirection[1] * gateHalfWidth, -gateDirection[0] * gateHalfWidth])
    gate = shapely.LineString([gateLeft, gateRight])

    return gate, gateMidpoint, gateDirection


def getGateExtendLine(gateMidpoint: NDArrayFloat1D,
                      gateDirection: NDArrayFloat1D,
                      leftExtendWidth: float,
                      rightExtendWidth: float) -> shapely.LineString:
    """
    Calculates the Shapely LineString of the gate defined by its midpoint,
    direction, width on the left, and width on the right. The gate is extended
    up to the extend limits.

    Args:
        gateMidpoint: Coordinates of the midpoint of the gate, in the form
            [x, y].
        gateDirection: Direction vector of the gate in the direction of forward
            travel, normalised to a magnitude of 1.
        leftExtendWidth: Distance from the gate midpoint to the left extend
            limits.
        rightExtendWidth: Distance from the gate midpoint to the right extend
            limits.

    Returns:
        Shapely LineString of the gate defined by gateMidpoint and
        gateDirection, with width on the left of leftExtendWidth and width on
        the right of rightExtendWidth.
    """
    extendLineLeft = gateMidpoint + ([-gateDirection[1] * leftExtendWidth, gateDirection[0] * leftExtendWidth])
    extendLineRight = gateMidpoint + ([gateDirection[1] * rightExtendWidth, -gateDirection[0] * rightExtendWidth])
    extendLine = shapely.LineString([extendLineLeft, extendLineRight])
    return extendLine


def getReducedLimitsClosed(limits: NDArrayFloat2D,
                           indexes: NDArrayInt1D,
                           nLimits: int,
                           distances: NDArrayFloat1D,
                           prevDist: float,
                           gateStep: float,
                           reducedWindow: float) -> tuple[NDArrayFloat2D, list[float]]:
    """
    Calculates the reduced limits and reduced cumulative distances, for the
    window specified.

    The reduced limits are the reduced set of limits coordinates local to the
    point. These are used to reduce the search space when finding the gate
    intersection with the track limits, and give robustness for handling complex
    figure-8 tracks.

    Args:
        limits: 2D array where each element is an [x, y, z] coordinate of the
            track limit, and each element increases the distance along the
            track.
        indexes: Array of the integer indexes corresponding to the limits array.
        nLimits: Number of limits coordinates (i.e. size of the limits array in
            axis 0).
        distances: 1D array representing the cumulative distance along the
            limits coordinates.
        prevDist: Distance along the limits where the previous gate intersected
            with the limits.
        gateStep: Distance between consecutive gate midpoints.
        reducedWindow: Distance ahead and behind the naive estimation of the
            gate intersection to include for the reduced limits.

    Returns:
        Tuple of (reducedLimitsCoords, reducedDist).

        reducedLimitsCoords: 2D array of [x, y, z] coordinates of the limits
        which were within the reducedWindow of the naive estimation of the gate
        intersection. This includes the coordinates on the boundary of the
        reducedWindow, calculated from linear interpolation.

        reducedDist: List representing the cumulative distance along the limits
        coordinates which were within the reducedWindow of the naive estimation
        of the gate intersection. This includes the cumulative distances on the
        boundary of the reducedWindow, calculated from linear interpolation.
    """
    # Sets the window to be length (2 * reducedWindow), centred around the point of expected intersection of the gate and limits
    distStart = utils.wrap(prevDist + gateStep - reducedWindow, 0, distances[-1])
    reducedDist = [distStart]
    reducedLimitsCoords = np.array([np.interp(distStart, distances, limits[:, 0]), np.interp(distStart, distances, limits[:, 1])])
    distStop = utils.wrap(prevDist + gateStep + reducedWindow, 0, distances[-1])

    # Finds index of the limits array to start searching from
    iStart = int(np.ceil(np.interp(distStart, distances, indexes)))
    i = utils.wrap(iStart + 1, 0, nLimits) if distances[iStart] == distStart else utils.wrap(iStart, 0, nLimits)

    # Iterate through the points in the limits coordinate array until the window is exceeded
    prevLeftDist = distStart
    passed = False
    while not passed:
        if prevLeftDist <= distStop <= distances[i] or distances[i] < prevLeftDist <= distStop:
            # If the window has been exceeded, calculate the point at the end of the window using linear interpolation
            passed = True
            reducedDist.append(distStop)
            reducedLimitsCoords = np.vstack((reducedLimitsCoords, [np.interp(distStop, distances, limits[:, 0]), np.interp(distStop, distances, limits[:, 1])]))
        else:
            # If the point is still within the window, append it
            reducedDist.append(distances[i])
            reducedLimitsCoords = np.vstack((reducedLimitsCoords, limits[i][0:2]))
            prevLeftDist = distances[i]
            i = utils.wrap(i + 1, 0, nLimits)

        # Terminate the while loop if it went the whole way around the track
        if i == iStart:
            print("getReducedLimits went around the track without passing the window, terminating loop")
            break

    return reducedLimitsCoords, reducedDist


def getLimitsExtendWidthClosed(gate: shapely.LineString,
                               gateMidpoint: NDArrayFloat1D,
                               limitsExtend: NDArrayFloat2D,
                               nLimitsExtend: float,
                               prevIndex: int,
                               gateHalfWidth: float) -> tuple[float, int]:
    """
    Calculates the distance to the extend limits from the gate midpoint, and the
    index from which the intersecting segment of the extend limits started.

    Args:
        gate: Shapely LineString representing the gate.
        gateMidpoint: Coordinates of the midpoint of the gate, in the form
            [x, y].
        limitsExtend: 2D array where each element is an [x, y, z] coordinate of
            the extend limit, and each element increases the distance along the
            track.
        nLimitsExtend: Number of extend limits coordinates (i.e. size of the
            limits array in axis 0).
        prevIndex: Index from which the segment of the extend limit which
            intersected with the previous gate started.
        gateHalfWidth: Half-width of the gate.

    Returns:
        Tuple of (extendWidth, prevIndex)

        extendWidth: Distance to the extend limits from the gate midpoint, along
        the width of the gate.

        prevIndex: Index of the coordinates from which the segment of the extend
        limits which intersected with the gate started.
    """
    gateMidpointPoint = shapely.Point(gateMidpoint)
    gateCoords = gate.xy

    if prevIndex < 0:
        # The first gate
        if utils.sideOfLine(limitsExtend[0][0], limitsExtend[0][1], gateCoords[0][0], gateCoords[1][0], gateCoords[0][1], gateCoords[1][0]) >= 0:
            # First extend point is after the gate, so iterate backwards from the last extend point since this is a closed circuit
            i = nLimitsExtend - 1
            iStep = -1
        else:
            # First extend point is before the gate, so iterate forwards from the first extend point
            i = 0
            iStep = 1
    else:
        # Normal procedure
        i = prevIndex
        iStep = 1

    # Iterate until an intersection is found (or if it went the whole way around the track)
    # Whether iteration is forwards or backwards is determined by iStep (1 is forwards, -1 is backwards)
    inLoop = False
    while not (inLoop and i == prevIndex):
        inLoop = True
        iNext = utils.wrap(i + iStep, 0, nLimitsExtend)
        extendSegment = shapely.LineString([limitsExtend[i][0:2], limitsExtend[iNext][0:2]])
        if extendSegment.intersects(gate):
            # If there is an intersection, update prevIndex and get the distance to the intersection
            prevIndex = i
            extendWidth = gateMidpointPoint.distance(extendSegment.intersection(gate))
            return extendWidth, prevIndex
        i = iNext

    # If the iterations went the whole way around the track without finding an intersection, return extendWidth=gateHalfWidth, prevIndex=prevIndex
    print("Didn't find an intersection with limitsExtend for gate at midpoint", gateMidpoint)
    return gateHalfWidth, prevIndex


def getLimitsWidths(gate: shapely.LineString,
                    gateMidpoint: NDArrayFloat1D,
                    reducedLeft: shapely.LineString,
                    reducedRight: shapely.LineString) -> tuple[float, float]:
    """
    Calculates the distance along the gate width to the left and right limits,
    from the midpoint of the gate.

    Args:
        gate: Shapely LineString representing the gate.
        gateMidpoint: Coordinates of the midpoint of the gate, in the form
            [x, y].
        reducedLeft: Shapely LineString connecting all the coordinates in the
            reduced left limits coordinate array, which is the reduced set of
            limits coordinates local to the point.
        reducedRight: Shapely LineString connecting all the coordinates in the
            reduced right limits coordinate array, which is the reduced set of
            limits coordinates local to the point.

    Returns:
        Tuple of (leftWidth, rightWidth)

        leftWidth: Distance to the left limits from the gate midpoint, along
        the width of the gate.

        rightWidth: Distance to the right limits from the gate midpoint, along
        the width of the gate.
    """
    gateMidpointPoint = shapely.Point(gateMidpoint)

    leftWidth = gateMidpointPoint.distance(reducedLeft.intersection(gate))
    rightWidth = gateMidpointPoint.distance(reducedRight.intersection(gate))

    return leftWidth, rightWidth


def calcGate(params: list[float],
             prevGateMidpoint: NDArrayFloat1D,
             prevGateDirection: NDArrayFloat1D,
             gateHalfWidth: float,
             gateStep: float,
             reducedLeft: shapely.LineString,
             reducedRight: shapely.LineString) -> tuple[shapely.LineString, NDArrayFloat1D, NDArrayFloat1D, float, float]:
    """
    Calculates the gate from the parameters used in the gate finding function,
    and information about the previous gate, then calculates the left and right
    widths to the track limits.

    Args:
        params: List of [psi, theta], in this format for compatibility with the
            SciPy optimisation used for gate finding.
            psi is the anti-clockwise angle from the previous gate
            direction, to place the midpoint of the new gate.
            theta is the direction of the new gate, as an anti-clockwise angle
            offset from psi.
        prevGateMidpoint: Coordinates of the midpoint of the previous gate, in
            the form [x, y].
        prevGateDirection: Direction vector of the previous gate in the
            direction of forward travel, normalised to a magnitude of 1.
        gateHalfWidth: Half-width of the gate.
        gateStep: Distance between consecutive gate midpoints.
        reducedLeft: Shapely LineString connecting all the coordinates in the
            reduced left limits coordinate array, which is the reduced set of
            limits coordinates local to the point.
        reducedRight: Shapely LineString connecting all the coordinates in the
            reduced right limits coordinate array, which is the reduced set of
            limits coordinates local to the point.

    Returns:
        Tuple of (gate, gateMidpoint, gateDirection, leftWidth, rightWidth).

        gate: Shapely LineString representing the gate.

        gateMidpoint: Coordinates of the midpoint of the gate, in the form
            [x, y].
        gateDirection: Direction vector of the gate in the direction of forward
            travel, normalised to a magnitude of 1.

        leftWidth: Distance to the left limits from the gate midpoint, along
        the width of the gate.

        rightWidth: Distance to the right limits from the gate midpoint, along
        the width of the gate.
    """
    psi = params[0]  # gateHeading (angle from previous gate direction, positive anti-clockwise)
    theta = params[1]  # gateAngle (angle from psi/gateHeading, positive anti-clockwise)

    # Generate candidate midpoint
    gateMidpoint = prevGateMidpoint + (utils.rotateVector2D(prevGateDirection, psi) * gateStep)

    # Generate candidate gate
    gateDirection = utils.rotateVector2D(prevGateDirection, psi + theta)
    gateLeft = gateMidpoint + ([-gateDirection[1] * gateHalfWidth, gateDirection[0] * gateHalfWidth])
    gateRight = gateMidpoint + ([gateDirection[1] * gateHalfWidth, -gateDirection[0] * gateHalfWidth])
    gate = shapely.LineString([gateLeft, gateRight])

    # Calculate leftWidth and rightWidth
    leftWidth, rightWidth = getLimitsWidths(gate, gateMidpoint, reducedLeft, reducedRight)

    # In case there was no intersection
    if np.isnan(leftWidth):
        leftWidth = gateHalfWidth
    if np.isnan(rightWidth):
        rightWidth = gateHalfWidth

    return gate, gateMidpoint, gateDirection, leftWidth, rightWidth


def gateObjFunc(params: list[float],
                prevGateMidpoint: NDArrayFloat1D,
                prevGateDirection: NDArrayFloat1D,
                gateHalfWidth: float,
                gateStep: float,
                reducedLeft: shapely.LineString,
                reducedRight: shapely.LineString) -> float:
    """
    Objective function for the gate finding optimisation.

    This objective function aims to find the gate placement to be in the middle
    of the track (i.e. equal distance to the left and right track limits), and
    with the smallest width to the left and right track limits.

    TODO: Explore using SciPy optimize root, with the objective function being
        a vector of [leftWidth - rightWidth, leftAngle - rightAngle],
        where leftAngle is the interior angle between the gate and the left
        track limits, and similarly for rightAngle. The angles must be on the
        same side of the gate.
        Root finding approach should be faster than minimise, but if root
        finding fails, fallback to SciPy minimise with this objective function.

    Args:
        params: List of [psi, theta], in this format for compatibility with the
            SciPy optimisation used for gate finding.
            psi is the anti-clockwise angle from the previous gate
            direction, to place the midpoint of the new gate.
            theta is the direction of the new gate, as an anti-clockwise angle
            offset from psi.
        prevGateMidpoint: Coordinates of the midpoint of the previous gate, in
            the form [x, y].
        prevGateDirection: Direction vector of the previous gate in the
            direction of forward travel, normalised to a magnitude of 1.
        gateHalfWidth: Half-width of the gate.
        gateStep: Distance between consecutive gate midpoints.
        reducedLeft: Shapely LineString connecting all the coordinates in the
            reduced left limits coordinate array, which is the reduced set of
            limits coordinates local to the point.
        reducedRight: Shapely LineString connecting all the coordinates in the
            reduced right limits coordinate array, which is the reduced set of
            limits coordinates local to the point.

    Returns:
        leftWidth + rightWidth + abs(leftWidth - rightWidth), where these are
        defined as below.

        leftWidth: Distance to the left limits from the gate midpoint, along
        the width of the gate.

        rightWidth: Distance to the right limits from the gate midpoint, along
        the width of the gate.
    """
    gate, gateMidpoint, gateDirection, leftWidth, rightWidth = calcGate(params, prevGateMidpoint, prevGateDirection, gateHalfWidth, gateStep, reducedLeft, reducedRight)
    if leftWidth == gateHalfWidth or rightWidth == gateHalfWidth:
        return 2 * gateHalfWidth + abs(leftWidth - rightWidth)
    else:
        return leftWidth + rightWidth + abs(leftWidth - rightWidth)


def getGateLimitsIntersectionDistance(gate: shapely.LineString,
                                      reducedLimitsCoords: NDArrayFloat2D,
                                      reducedDist: list[float],
                                      distances: NDArrayFloat1D,
                                      isLeft: bool) -> float:
    """
    Calculates the distance along the track limits of the intersection with the
    gate.

    Args:
        gate: Shapely LineString representing the gate.
        reducedLimitsCoords: 2D array of [x, y, z] coordinates of the limits
            which were within the reducedWindow of the naive estimation of the
            gate intersection. This includes the coordinates on the boundary of
            the reducedWindow, calculated from linear interpolation.
        reducedDist: List representing the cumulative distance along the limits
            coordinates which were within the reducedWindow of the naive
            estimation of the gate intersection. This includes the cumulative
            distances on the boundary of the reducedWindow, calculated from
            linear interpolation.

    Returns:
        Distance along the coordinate array of the intersection with the gate.
    """
    # Iterates through each segment in the reduced limits coordinate array
    for i in range(len(reducedDist) - 1):
        # Creates a segment from adjacent points in the reduced limits coordinate array and checks if the segment intersects the gate
        segment = shapely.LineString([reducedLimitsCoords[i], reducedLimitsCoords[i + 1]])
        if segment.intersects(gate):
            # If there is an intersection, get the intersection point (note this can technically return a LineString but that's super unlikely)
            intersection = segment.intersection(gate)

            # Unwrap distances so the interpolation works properly (only applies near the start line if the track is closed)
            d1 = reducedDist[i]
            d2 = reducedDist[i + 1]
            if d1 > d2:
                d1 -= distances[-1]

            # Chooses axis with larger difference to do the linear interpolation on to find the distance along the limits coordinate array of intersection
            if np.diff(segment.xy[0]) > np.diff(segment.xy[1]):
                dist = np.interp(intersection.x, segment.xy[0], [d1, d2])
            else:
                dist = np.interp(intersection.y, segment.xy[1], [d1, d2])

            return dist

    # If the gate never intersected with reducedLimitsCoords, return the first distance in the distances array,
    # and the gate's x and y coordinates on the side corresponding to isLeft
    if isLeft:
        print("leftWidth == gateHalfWidth for gate", gate)
        return reducedDist[0]
    else:
        print("rightWidth == gateHalfWidth for gate", gate)
        return reducedDist[0]


class Track:
    def __init__(self,
                 left: list[list[float]] | NDArrayFloat2D,
                 right: list[list[float]] | NDArrayFloat2D,
                 leftExtend: list[list[float]] | NDArrayFloat2D | None = None,
                 rightExtend: list[list[float]] | NDArrayFloat2D | None = None,
                 startLineCoords: list[list[float]] | NDArrayFloat2D | None = None,
                 finishLineCoords: list[list[float]] | NDArrayFloat2D | None = None,
                 isClosed: bool = None,
                 gateStep: float = 10) -> None:
        print("Initialising track")

        # Constants (subject to change though) TODO: Consider moving all settings to a separate Python file, grouping them by module
        isClosedTrackThreshold = 10     # Threshold gap size in metres for if the left/right track limits coordinate arrays provided are closed or not
        gateHalfWidth = 50              # Half-width of the gate in metres (only used during
        reducedWindow = 2 * gateStep    # Distance window for the reduced track left/right lists when creating gates

        # Not sure if this will be needed in the future but currently used to not unnecessarily duplicate points in the track z height interpolators
        leftExtendProvided = leftExtend is not None
        rightExtendProvided = rightExtend is not None

        # Convert relevant inputs to NumPy arrays and set leftExtend and rightExtend to left and right respectively if they're None
        left = np.array(left)
        right = np.array(right)
        leftExtend = np.array(leftExtend) if leftExtendProvided else left
        rightExtend = np.array(rightExtend) if rightExtendProvided else right
        startLineCoords = np.array(startLineCoords) if startLineCoords else None
        finishLineCoords = np.array(finishLineCoords) if finishLineCoords else None

        # Get min/max x and y coordinates
        xCoords = np.concat((left[:, 0], right[:, 0], leftExtend[:, 0], rightExtend[:, 0]))
        yCoords = np.concat((left[:, 1], right[:, 1], leftExtend[:, 1], rightExtend[:, 1]))
        self.xMin = np.min(xCoords)
        self.xMax = np.max(xCoords)
        self.yMin = np.min(yCoords)
        self.yMax = np.max(yCoords)

        # Creates track z height interpolators from all provided [x, y, z] coordinates - left, right, leftExtend and rightExtend arrays
        # Create data point coordinate array ([x, y] coordinates)
        coords = np.vstack((left[:, :2].copy(), right[:, :2].copy()))
        if leftExtendProvided:
            coords = np.vstack((coords, leftExtend[:, :2].copy()))
        if rightExtendProvided:
            coords = np.vstack((coords, rightExtend[:, :2].copy()))
        # Create data value array (z values)
        zValues = np.hstack((left[:, 2].copy(), right[:, 2].copy()))
        if leftExtendProvided:
            zValues = np.hstack((zValues, leftExtend[:, 2].copy()))
        if rightExtendProvided:
            zValues = np.hstack((zValues, rightExtend[:, 2].copy()))
        # Create interpolators
        self.zLinInterp = scipy.interpolate.LinearNDInterpolator(coords, zValues)
        self.zNNInterp = scipy.interpolate.NearestNDInterpolator(coords, zValues)

        # Determine whether the track provided is closed based on left and right arrays if not available
        if isClosed is None:
            gapLeft = scipy.linalg.norm(left[0] - left[-1])
            gapRight = scipy.linalg.norm(right[0] - right[-1])
            if max(gapLeft, gapRight) < isClosedTrackThreshold:
                self.isClosed = True
            else:
                self.isClosed = False
        else:
            self.isClosed = isClosed

        # Make sure left, right, leftExtend and rightExtend arrays are closed if the track is closed
        if self.isClosed:
            if not np.array_equal(left[0], left[-1]):
                left = np.vstack((left, left[0]))
            if not np.array_equal(right[0], right[-1]):
                right = np.vstack((right, right[0]))
            if not np.array_equal(leftExtend[0], leftExtend[-1]):
                leftExtend = np.vstack((leftExtend, leftExtend[0]))
            if not np.array_equal(rightExtend[0], rightExtend[-1]):
                rightExtend = np.vstack((rightExtend, rightExtend[0]))

        # Create start/finish gate coordinates if they aren't available, based on the left and right arrays
        if startLineCoords is None:
            startLineCoords = finishLineCoords if self.isClosed and finishLineCoords else np.array([left[0][:2], right[0][:2]])
        if finishLineCoords is None:
            finishLineCoords = startLineCoords if self.isClosed else np.array([left[-1][:2], right[-1][:2]])

        # Create startLine/finishLine lines from their coordinates - to determine when to insert the actual startGate and finishGate into the gates
        startLine = shapely.LineString(startLineCoords)
        finishLine = shapely.LineString(finishLineCoords)

        # Lists for gates and their related data - these will be turned into NumPy arrays at the end
        self.gates = []
        self.gatesMidpoint = []
        self.gatesDirection = []
        self.leftWidths = []
        self.rightWidths = []
        self.leftExtendWidths = []
        self.rightExtendWidths = []

        # Indexes of the start and finish gates in the gates arrays
        self.startGateIndex = -1
        self.finishGateIndex = -1

        # Create arrays storing the distances along the left/right track limits
        nLeft, leftDistances = getLimitsDistances(left)
        nRight, rightDistances = getLimitsDistances(right)
        nLeft = np.size(leftDistances)
        nLeft = np.size(leftDistances)

        # For the extendWidth calculations
        nLeftExtend = np.size(leftExtend, 0)
        nRightExtend = np.size(rightExtend, 0)
        prevLeftExtendIndex = -1
        prevRightExtendIndex = -1

        # Create the first gate and its related data, then append it to the relevant lists
        gate, gateMidpoint, gateDirection = getGateFromCoords(left[0][:2], right[0][:2], gateHalfWidth)
        leftWidth = scipy.linalg.norm(gateMidpoint - left[0][0:2])
        rightWidth = scipy.linalg.norm(gateMidpoint - right[0][0:2])
        leftExtendWidth, prevLeftExtendIndex = getLimitsExtendWidthClosed(gate, gateMidpoint, leftExtend, nLeftExtend, prevLeftExtendIndex, gateHalfWidth)
        rightExtendWidth, prevRightExtendIndex = getLimitsExtendWidthClosed(gate, gateMidpoint, rightExtend, nRightExtend, prevRightExtendIndex, gateHalfWidth)
        self.gates.append(gate)
        self.gatesMidpoint.append(gateMidpoint)
        self.gatesDirection.append(gateDirection)
        self.leftWidths.append(leftWidth)
        self.rightWidths.append(rightWidth)
        self.leftExtendWidths.append(max(leftExtendWidth, leftWidth))
        self.rightExtendWidths.append(max(rightExtendWidth, rightWidth))

        # Create the last gate - for detecting when to stop gate creation so this doesn't have the related data and is only within track limits
        lastGate = shapely.LineString([left[-1][:2], right[-1][:2]])

        # Calculate subsequent gates - loop stops once we've gone around the whole track (also stops/throws an error if gate creation breaks)
        print("Creating track gates")
        leftIndexes = np.arange(nLeft)
        rightIndexes = np.arange(nRight)
        prevLeftDist = 0
        prevRightDist = 0
        while True:
            # Create reducedLeftCoords and reducedRightCoords coordinate lists
            if self.isClosed:
                reducedLeftCoords, reducedLeftDist = getReducedLimitsClosed(left, leftIndexes, nLeft, leftDistances, prevLeftDist, gateStep, reducedWindow)
                reducedRightCoords, reducedRightDist = getReducedLimitsClosed(right, rightIndexes, nRight, rightDistances, prevRightDist, gateStep, reducedWindow)
            else:
                raise Exception("No logic for if track is not closed - pls fix")
                # Should be similar to if isClosed but instead of wrapping it clips to the min and max

            # Create reducedLeft and reducedRight LineStrings
            reducedLeft = shapely.LineString(reducedLeftCoords)
            reducedRight = shapely.LineString(reducedRightCoords)

            # Find gate heading and direction
            params = scipy.optimize.minimize(gateObjFunc, [0, 0],
                                             (gateMidpoint, gateDirection, gateHalfWidth, gateStep, reducedLeft, reducedRight),
                                             method='Powell').x
            # Experiment with different scipy minimize methods to see which is faster and also accuracy - ones that solved successfully:
            #   'Nelder-Mead'   20.256154368287984
            #   'Powell'        20.254838726180974
            #   'L-BFGS-B'      20.25543032184799
            #   'TNC'           20.261686355342427
            #   'COBYLA'        FAILED
            #   'COBYQA'        20.27994131907119
            #   'SLSQP'         20.252698788205983
            #   'trust-constr'  20.258087401969213

            gate, gateMidpoint, gateDirection, leftWidth, rightWidth = calcGate(params, gateMidpoint, gateDirection, gateHalfWidth, gateStep, reducedLeft, reducedRight)

            # Raise an exception if gate creation explodes (gateMidpoint goes beyond the bounds of xMin, xMax, yMin, yMax)
            if gateMidpoint[0] < self.xMin or gateMidpoint[0] > self.xMax or gateMidpoint[1] < self.yMin or gateMidpoint[1] > self.yMax or leftWidth + rightWidth == 2 * gateHalfWidth:
                raise Exception("Gate creation exploded - Gate with gateMidpoint " + str(gateMidpoint.tolist()) + " went beyond the bounds of [(xMin, xMax), (yMin, yMax)] of " + str([(float(self.xMin), float(self.xMax)), (float(self.yMin), float(self.yMax))]))

            # Find the gate intersections with the left and right coordinate arrays
            prevLeftDist = getGateLimitsIntersectionDistance(gate, reducedLeftCoords, reducedLeftDist, leftDistances, True)
            prevRightDist = getGateLimitsIntersectionDistance(gate, reducedRightCoords, reducedRightDist, rightDistances, False)

            # Find leftExtendWidth and rightExtendWidth - using candidates for the indexes to make sure the indexes only update if the gate was appended to the lists
            leftExtendWidth, prevLeftExtendIndexCandidate = getLimitsExtendWidthClosed(gate, gateMidpoint, leftExtend, nLeftExtend, prevLeftExtendIndex, gateHalfWidth)
            rightExtendWidth, prevRightExtendIndexCandidate = getLimitsExtendWidthClosed(gate, gateMidpoint, rightExtend, nRightExtend, prevRightExtendIndex, gateHalfWidth)

            # Create a segment from the previous gateMidpoint to the current gateMidpoint - to check intersections with startLine, finishLine and last gate
            midlineSegment = shapely.LineString([self.gatesMidpoint[-1], gateMidpoint])

            # Check intersection with startLine
            if midlineSegment.intersects(startLine) and self.startGateIndex < 0:
                # Check that startGate isn't the previous gate or the current gate (should only happen if startCoords were automatically computed)
                startGate, startGateMidpoint, startGateDirection = getGateFromCoords(startLineCoords[0], startLineCoords[1], gateHalfWidth)
                if shapely.equals_exact(startGate, self.gates[-1]):
                    # startGate is the previous gate
                    self.startGateIndex = len(self.gates) - 1
                elif shapely.equals_exact(startGate, gate):
                    # startGate is the current gate
                    self.startGateIndex = len(self.gates)
                else:
                    # Create start gate from coordinates then calculate all the gate information and append to the lists
                    startGate, startGateMidpoint, startGateDirection = getGateFromCoords(startLineCoords[0], startLineCoords[1], gateHalfWidth)
                    startLeftWidth, startRightWidth = getLimitsWidths(startGate, startGateMidpoint, reducedLeft, reducedRight)
                    startLeftExtendWidth, prevLeftExtendIndex = getLimitsExtendWidthClosed(startGate, startGateMidpoint, leftExtend, nLeftExtend, prevLeftExtendIndex, gateHalfWidth)
                    startRightExtendWidth, prevRightExtendIndex = getLimitsExtendWidthClosed(startGate, startGateMidpoint, rightExtend, nRightExtend, prevRightExtendIndex, gateHalfWidth)
                    # Iterate backwards and pop previous gates that intersect with the startGate within the extend limits
                    startGateExtendLine = getGateExtendLine(startGateMidpoint, startGateDirection, startLeftExtendWidth, startRightExtendWidth)
                    while len(self.gates) >= 1:
                        if startGateExtendLine.intersects(getGateExtendLine(self.gatesMidpoint[-1], self.gatesDirection[-1], self.leftExtendWidths[-1], self.rightExtendWidths[-1])):
                            print("Gate at midpoint", self.gatesMidpoint[-1].tolist(), "and startGate both have gateExtendLines that intersect - removing this gate from the lists")
                            self.gates.pop()
                            self.gatesMidpoint.pop()
                            self.gatesDirection.pop()
                            self.leftWidths.pop()
                            self.rightWidths.pop()
                            self.leftExtendWidths.pop()
                            self.rightExtendWidths.pop()
                        else:
                            break
                    self.startGateIndex = len(self.gates)
                    self.gates.append(startGate)
                    self.gatesMidpoint.append(startGateMidpoint)
                    self.gatesDirection.append(startGateDirection)
                    self.leftWidths.append(startLeftWidth)
                    self.rightWidths.append(startRightWidth)
                    self.leftExtendWidths.append(max(startLeftExtendWidth, startLeftWidth))
                    self.rightExtendWidths.append(max(startRightExtendWidth, startRightWidth))
                print("Start gate index:", self.startGateIndex)

            # Check intersection with finishLine - Note if finishLine is unique but very close to startLine then finishGateIndex = startGateIndex + 1
            if midlineSegment.intersects(finishLine) and self.finishGateIndex < 0:
                # Check that finishGate isn't the previous gate or the current gate (should only happen if startCoords were automatically computed)
                finishGate, finishGateMidpoint, finishGateDirection = getGateFromCoords(finishLineCoords[0], finishLineCoords[1], gateHalfWidth)
                if shapely.equals_exact(finishGate, self.gates[-1]):
                    # finishGate is the previous gate
                    self.finishGateIndex = len(self.gates) - 1
                elif shapely.equals_exact(finishGate, gate):
                    # finishGate is the current gate, Check that finishLine isn't also startLine (very rare possibility if self.isClosed)
                    if shapely.equals_exact(startLine, finishLine):
                        # finishLine is also startLine, which means we've already got it in the gates lists from the startLine intersection check
                        self.finishGateIndex = self.startGateIndex
                    else:
                        self.finishGateIndex = len(self.gates)
                else:
                    # Create finish gate from coordinates then calculate all the gate information and append to the lists
                    finishGate, finishGateMidpoint, finishGateDirection = getGateFromCoords(finishLineCoords[0], finishLineCoords[1], gateHalfWidth)
                    finishLeftWidth, finishRightWidth = getLimitsWidths(finishGate, finishGateMidpoint, reducedLeft, reducedRight)
                    finishLeftExtendWidth, prevLeftExtendIndex = getLimitsExtendWidthClosed(finishGate, finishGateMidpoint, leftExtend, nLeftExtend, prevLeftExtendIndex, gateHalfWidth)
                    finishRightExtendWidth, prevRightExtendIndex = getLimitsExtendWidthClosed(finishGate, finishGateMidpoint, rightExtend, nRightExtend, prevRightExtendIndex, gateHalfWidth)
                    # Iterate backwards and pop previous gates that intersect with the finishGate within the extend limits
                    finishGateExtendLine = getGateExtendLine(finishGateMidpoint, finishGateDirection, finishLeftExtendWidth, finishRightExtendWidth)
                    while len(self.gates) >= 1:
                        if finishGateExtendLine.intersects(getGateExtendLine(self.gatesMidpoint[-1], self.gatesDirection[-1], self.leftExtendWidths[-1], self.rightExtendWidths[-1])):
                            print("Gate at midpoint", self.gatesMidpoint[-1].tolist(), "and finishGate both have gateExtendLines that intersect - removing this gate from the lists")
                            self.gates.pop()
                            self.gatesMidpoint.pop()
                            self.gatesDirection.pop()
                            self.leftWidths.pop()
                            self.rightWidths.pop()
                            self.leftExtendWidths.pop()
                            self.rightExtendWidths.pop()
                        else:
                            break
                    self.finishGateIndex = len(self.gates)
                    self.gates.append(finishGate)
                    self.gatesMidpoint.append(finishGateMidpoint)
                    self.gatesDirection.append(finishGateDirection)
                    self.leftWidths.append(finishLeftWidth)
                    self.rightWidths.append(finishRightWidth)
                    self.leftExtendWidths.append(max(finishLeftExtendWidth, finishLeftWidth))
                    self.rightExtendWidths.append(max(finishRightExtendWidth, finishRightWidth))
                print("Finish gate index:", self.finishGateIndex)

            # Check intersection with lastGate - must also be 4 or more gates in the list to break out of the gate creation loop
            if midlineSegment.intersects(lastGate) and len(self.gates) >= 4:
                # If self.isClosed then this gate will be the same as the first gate so it's unnecessary to add the last gate
                # If track is not closed, re-make the last gate from coordinates and calculate all the gate information and append to lists
                if not self.isClosed:
                    lastGate, lastGateMidpoint, lastGateDirection = getGateFromCoords(left[-1][:2], right[-1][:2], gateHalfWidth)
                    lastLeftWidth = scipy.linalg.norm(lastGateMidpoint - left[-1][0:2])
                    lastRightWidth = scipy.linalg.norm(lastGateMidpoint - right[-1][0:2])
                    lastLeftExtendWidth, prevLeftExtendIndex = getLimitsExtendWidthClosed(lastGate, lastGateMidpoint, leftExtend, nLeftExtend, prevLeftExtendIndex, gateHalfWidth)
                    lastRightExtendWidth, prevRightExtendIndex = getLimitsExtendWidthClosed(lastGate, lastGateMidpoint, rightExtend, nRightExtend, prevRightExtendIndex, gateHalfWidth)
                    # Iterate backwards and pop previous gates that intersect with the lastGate within the extend limits
                    lastGateExtendLine = getGateExtendLine(lastGateMidpoint, lastGateDirection, lastLeftExtendWidth, lastRightExtendWidth)
                    while len(self.gates) >= 1:
                        if lastGateExtendLine.intersects(getGateExtendLine(self.gatesMidpoint[-1], self.gatesDirection[-1], self.leftExtendWidths[-1], self.rightExtendWidths[-1])):
                            print("Gate at midpoint", self.gatesMidpoint[-1].tolist(), "and lastGate both have gateExtendLines that intersect - removing this gate from the lists")
                            self.gates.pop()
                            self.gatesMidpoint.pop()
                            self.gatesDirection.pop()
                            self.leftWidths.pop()
                            self.rightWidths.pop()
                            self.leftExtendWidths.pop()
                            self.rightExtendWidths.pop()
                        else:
                            break
                    self.gates.append(lastGate)
                    self.gatesMidpoint.append(lastGateMidpoint)
                    self.gatesDirection.append(lastGateDirection)
                    self.leftWidths.append(lastLeftWidth)
                    self.rightWidths.append(lastRightWidth)
                    self.leftExtendWidths.append(max(lastLeftExtendWidth, lastLeftWidth))
                    self.rightExtendWidths.append(max(lastRightExtendWidth, lastRightWidth))
                # Exit the gate creation loop
                print("Finished gate creation - Total number of gates:", len(self.gates))
                break

            # Check if this gate intersects with the previous gate within the extend limits
            if getGateExtendLine(gateMidpoint, gateDirection, leftExtendWidth, rightExtendWidth).intersects(getGateExtendLine(self.gatesMidpoint[-1], self.gatesDirection[-1], self.leftExtendWidths[-1], self.rightExtendWidths[-1])):
                print("Gate at midpoint", gateMidpoint.tolist(), "and previous gate both have gateExtendLines that intersect - not appending this gate to the lists")
            else:
                # If this gate doesn't intersect then append the gate and its information
                self.gates.append(gate)
                self.gatesMidpoint.append(gateMidpoint)
                self.gatesDirection.append(gateDirection)
                self.leftWidths.append(leftWidth)
                self.rightWidths.append(rightWidth)
                self.leftExtendWidths.append(max(leftExtendWidth, leftWidth))
                self.rightExtendWidths.append(max(rightExtendWidth, rightWidth))
                prevLeftExtendIndex = prevLeftExtendIndexCandidate
                prevRightExtendIndex = prevRightExtendIndexCandidate

        # Raise an exception if startLine or finishLine wasn't crossed when creating the gates
        if self.startGateIndex < 0:
            raise Exception("Start line was not crossed during gate creation")
        if self.finishGateIndex < 0:
            raise Exception("Finish line was not crossed during gate creation")

        self.gates = np.array(self.gates)
        self.gatesMidpoint = np.array(self.gatesMidpoint)
        self.gatesDirection = np.array(self.gatesDirection)
        self.leftWidths = np.array(self.leftWidths)
        self.rightWidths = np.array(self.rightWidths)
        self.leftExtendWidths = np.array(self.leftExtendWidths)
        self.rightExtendWidths = np.array(self.rightExtendWidths)

        # Track initialised :)
        print("Track initialised")


    def getZ(self,
             x: float,
             y: float) -> float:
        """
        Calculates the z coordinate (height) of the track at the input x and y coordinates.

        Currently this is a very simple implementation using the linear interpolation maps.
        TODO: Robust approach for compatibility with figure-8 tracks like Suzuka. Possible handling for complex tracks is using multiple interpolators
            for each track segment in the local region around each gate (will need logic to define a sufficiently large local region to accommodate
            arbitrarily long wheelbases but not include "overlapping" track segments if the track is an upwards spiral). Adding an additional argument
            to determine the most recently passed gate (i.e. which interpolator to use).

        Args:
            x: x coordinate of the point.
            y: y coordinate of the point.

        Returns:
            z coordinate of the track for the point (x, y).
        """
        # Try linear interpolation first
        z = self.zLinInterp(x, y)

        # Check if linear interpolation was successful - if failed then do nearest neighbour interpolation
        if np.isnan(z):
            z = self.zNNInterp(x, y)

        return z

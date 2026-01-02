"""
The track module is responsible for defining the track on which a trajectory
can be created and optimised.

This includes the Track class, as well as the CoordinateArray, Event and Gate
classes used to generate and define the track.
"""
# Python standard libraries
from typing import Literal
from dataclasses import dataclass

# External libraries
import numpy as np
import shapely

# Project python modules
from Utils.typeAliases import Any, ListFloat2D, NDArrayFloat1D, NDArrayFloat2D, NDArrayInt1D, NDArrayNumber1D
import Utils.utils as utils

# Filename constants
TRACK_PKL_FILENAME = "Track.pkl"
LIMIT_LEFT_SOFT_FILENAME = "xyzLimitLeftSoft.csv"
LIMIT_RIGHT_SOFT_FILENAME = "xyzLimitRightSoft.csv"
LIMIT_LEFT_HARD_FILENAME = "xyzLimitLeftHard.csv"
LIMIT_RIGHT_HARD_FILENAME = "xyzLimitRightHard.csv"

# CoordinateArray constants
RESAMPLE_SPATIAL_FREQ = 1                       # Frequency to resample the coordinate array for low-pass filtering (cycles/m)
LP_FILT_SPATIAL_FREQ = 0.1                      # Low-pass cutoff spatial frequency (cycles/m)
LP_FILT_ORDER = 1                               # Order of the low-pass filter

# Event constants
CUSTOM_EVENT_TYPES = Literal['TODO']            # List of valid event types for custom events TODO: Implement custom events and add them here
INTERNAL_EVENT_TYPES = Literal['GateCreation',  # Valid event types for internal events
                               'StartFinish']

# Gate constants
GATE_MAX_WIDTH = 1000                           # Maximum width of the gate, used as the initial gate width during gate creation

GATE_EXTEND_WIDTH = 10                          # Additional width beyond the respective hard track limits to extend the gate width on each side, to
                                                # allow suitable penalties to be calculated if track limits are violated
                                                #  - Higher gives more robustness, allowing a trajectory that exceeds track limits to still be solved
                                                #  - Too high can cause gates to be skipped in tight corners due to overlapping, losing track limits
                                                #    resolution

# Track generation constants
CLOSED_TRACK_THRESHOLD_DISTANCE = 10            # Maximum distance (as the crow flies) from the start to finish coordinates of the provided soft track
                                                # limits to consider the track closed, when using the automatic logic

DIRECTION_SIMILARITY_THRESHOLD = 0              # Threshold for the dot product of 2 vectors (each with magnitude of 1) above which to consider the
                                                # directions of the vectors as "similar"

GATE_STEP_DISTANCE = 5                          # Distance between each consecutive gate for gate creation (unless overridden by an event gate or the
                                                # gate is skipped as it overlaps with neighbouring gates)
                                                #  - Higher gives more resolution for determining track limits
                                                #  - Too high may excessively slow track and trajectory generation

REDUCED_DISTANCE_WINDOW = 250                   # Window on each side for the distance bounds of the reduced coordinate array for gate creation
REDUCED_HEADING_WINDOW = np.pi                  # Window on each side for the heading angle bounds of the reduced coordinate array for gate creation

# Track plot constants
TRACK_PLOT_AREA = 100                           # Area of the track plot saved after track generation or if track generation raised an exception
                                                # manually (in square inches)

TRACK_PLOT_SPATIAL_RESOLUTION = 5               # Spatial resolution of the saved track plot (in pixels/metre)


class CoordinateArray:
    """
    Array of coordinates and its derived attributes.

    Used for track generation to store the coordinates defining the track limits
    and track mesh.

    Has methods:
        - getReducedCoordArray(sRef, sLower, sUpper, ALower, AUpper)

    Attributes:
        xyzCoords: Coordinates in order of increasing distance along the
            coordinate array, where each coordinate is in the form [x, y, z].
            2D array of floats.
        sCoords: Cumulative distance along the coordinates, assuming straight
            lines between coordinates. Starts from 0 if this is the "full"
            coordinate array, but can start at any value if this is a "reduced"
            coordinate array. 1D array of floats.
        AHeadings: Unwrapped heading angle along the coordinates, in radians. 0
            corresponds to positive y (north) and increasing clockwise. 1D array
            of floats.
        AHeadingsFilt: Low-pass filtered AHeadings, with cutoff frequency
            LP_FILT_SPATIAL_FREQ and filter order LP_FILT_ORDER. 1D array of
            floats.
    """
    def __init__(self,
                 xyzCoords: NDArrayFloat2D,
                 BClosedTrack: bool,
                 BAllowNegativeInitAHeading: bool,
                 sCoords: NDArrayFloat1D | None = None,
                 AHeadings: NDArrayFloat1D | None = None,
                 AHeadingsFilt: NDArrayFloat1D | None = None) -> None:
        """
        Initialises the coordinate array and calculates all of its attributes.

        Makes sure that the coordinates are closed if the track is closed,
        removes consecutive duplicate coordinates, calculates the cumulative
        distance along the coordinates, then calculates the raw and filtered
        unwrapped heading angle along the coordinates. If all attributes are
        provided, uses those (without doing any validation).

        Args:
            xyzCoords: Array of coordinates in order of increasing distance
                along the track, where each coordinate is in the form [x, y, z].
            BClosedTrack: Whether the coordinate array should be treated as
                closed. A closed coordinate array means that the last coordinate
                is equal to the first coordinate.
            BAllowNegativeInitAHeading: Whether to offset the initial heading
                angle (both raw and unfiltered) by 2 pi to ensure that the
                initial filtered heading angle is between -pi and pi.
            sCoords: If all of sCoords, AHeadings and AHeadingsFilt are
                provided, overrides the automatically calculated attribute
                sCoords.
            AHeadings: If all of sCoords, AHeadings and AHeadingsFilt are
                provided, overrides the automatically calculated attribute
                AHeadings.
            AHeadingsFilt: If all of sCoords, AHeadings and AHeadingsFilt are
                provided, overrides the automatically calculated attribute
                AHeadingsFilt.
        """
        # Set xyzCoords, making sure this is a closed coordinate array if the track is closed (last coordinate equal to the first coordinate)
        if BClosedTrack and xyzCoords[-1] != xyzCoords[0]:
            self.xyzCoords = np.vstack((xyzCoords, xyzCoords[0]))
        else:
            self.xyzCoords = utils.removeConsecutiveDuplicates(xyzCoords, axis=0)

        # Get the indexes that omit consecutive duplicates from the coordinate array, and remove the consecutive duplicates from the coordinate array
        inds = utils.getIndsWithoutConsecutiveDuplicates(xyzCoords, axis=0)
        self.xyzCoords = self.xyzCoords[inds]

        # Check if all the override attributes are provided
        if all(a is not None for a in [sCoords, AHeadings, AHeadingsFilt]):
            # All attributes provided, set all attributes from the overrides
            # Omit the same indexes which caused consecutive duplicates in the coordinate array
            self.sCoords = sCoords[inds]
            self.AHeadings = AHeadings[inds]
            self.AHeadingsFilt = AHeadingsFilt[inds]
        else:
            # Missing attributes, derive all attributes
            # Calculate sCoords - cumulative distance along the coordinates
            dxyzCoords = np.diff(xyzCoords, axis=0)
            dsCoords = np.linalg.norm(dxyzCoords, axis=1)
            self.sCoords = np.append(0, np.cumsum(dsCoords))

            # Calculate AHeadings - unwrapped heading angle along the coordinates, calculated as the average of the forward and backward angles
            AHeadingsTemp = np.unwrap(utils.getHeading(dxyzCoords))
            AHeadingsForwards = np.append(AHeadingsTemp[0], AHeadingsTemp)
            AHeadingsBackwards = np.append(AHeadingsTemp, AHeadingsTemp[0])
            self.AHeadings = (AHeadingsForwards + AHeadingsBackwards) / 2

            # Calculate AHeadingsFilt - low-pass filtered AHeadings
            # Resample to regular intervals, then filter, then resample back to original signal base
            AHeadingsResampled, sCoordsResampled = utils.resample(self.AHeadings, self.sCoords, RESAMPLE_SPATIAL_FREQ, True)
            AHeadingsFiltResampled = utils.filt(AHeadingsResampled, RESAMPLE_SPATIAL_FREQ, 'low', LP_FILT_SPATIAL_FREQ, LP_FILT_ORDER)
            self.AHeadingsFilt = np.interp(self.sCoords, sCoordsResampled, AHeadingsFiltResampled)

            # If BAllowNegativeInitHeading, offset AHeading and AHeadingFilt by 2 pi if necessary such that
            # the initial AHeadingFilt is in the range from -pi to pi
            if BAllowNegativeInitAHeading and self.AHeadingsFilt[0] > np.pi:
                self.AHeadings -= np.pi
                self.AHeadingsFilt -= np.pi

    def getReducedCoordArray(self,
                             sRef: float,
                             sLower: float,
                             sUpper: float,
                             ALower: float,
                             AUpper: float) -> CoordinateArray:
        """
        Get a reduced version of the CoordinateArray object which only contains
        the coordinates and attributes local to the region specified.

        The reduced coordinates are all the coordinates moving forward and
        backward from the reference distance sRef along the coordinate array,
        until it gets to a distance or heading bound.

        Args:
            sRef: Reference distance along the coordinate array. If the track is
                not closed, must be within the sLower and sUpper distance
                bounds.
            sLower: Lower distance bound. If the track is closed, can be
                provided wrapped or unwrapped (with the exception being the
                bounds would wrap to enclose the entire track, in which case,
                must be provided unwrapped for the expected behaviour). If the
                track is not closed, must be <= sRef.
            sUpper: Lower distance bound. If the track is closed, can be
                provided wrapped or unwrapped (with the exception being the
                bounds would wrap to enclose the entire track, in which case,
                must be provided unwrapped for the expected behaviour). If the
                track is not closed, must be <= sRef.
            ALower: Lower unwrapped heading angle bound.
            AUpper: Upper unwrapped heading angle bound.

        Returns:
            Reduced coordinate array, as a CoordinateArray object.

        Raises:
            ValueError: Invalid reference distance or heading angle bounds:
                heading angle at sRef {ARef} is outside the heading angle bounds
                [ALower, AUpper]
        """
        # Validate that the coordinate at the reference distance is within the heading angle bounds
        ARef = np.interp(sRef, self.sCoords, self.AHeadingsFilt)
        if not ALower <= ARef <= AUpper:
            raise ValueError(f"Invalid reference distance or heading angle bounds: "
                             f"heading angle at sRef {ARef} is outside the heading angle bounds [{ALower}, {AUpper}]")

        # Automatically determine if the track is closed
        BClosedTrack = all(self.xyzCoords[-1] == self.xyzCoords[0])

        def __getIndsDistance() -> NDArrayInt1D:
            """
            Internal function to get all the indexes of the coordinate array
            within the distance bounds.

            For closed tracks, the distance bounds can be either wrapped or
            unwrapped (their wrapped values are calculated and used internally),
            and computes "within the distance bounds" in the correct direction.

            Returns:
                All the indexes of the coordinate array within the distance
                bounds [sLower, sUpper], in order as the self.sCoords array.
            """
            if BClosedTrack and (sLower < 0 or sUpper > self.sCoords[-1] or sLower > sUpper):
                # Closed track and the bounds will wrap, or closed track and the bounds are already wrapped
                # Wrap the lower and upper distance bounds
                sLowerWrapped = utils.wrap(sLower, 0, self.sCoords[-1])
                sUpperWrapped = utils.wrap(sUpper, 0, self.sCoords[-1])

                # Check if the wrapped bounds cover the whole track
                if sLowerWrapped <= sUpperWrapped:
                    # Wrapped bounds cover the whole track, return all the indexes of the coordinate array
                    indsDistance = np.ndindex(len(self.sCoords))
                else:
                    # Wrapped bounds don't cover the whole track - and due to the wrapping, sLowerWrapped > sUpperWrapped
                    indsDistance = np.where((sLowerWrapped <= self.sCoords) & (self.sCoords <= sUpperWrapped))

            else:
                # Not a closed track or the bounds don't wrap, get the indexes where the coordinate array is within the distance bounds
                indsDistance = np.where(sLower <= self.sCoords <= sUpper)[0]

            return indsDistance

        def __getIndsHeading() -> NDArrayInt1D:
            """
            Internal function to get the indexes of the coordinate array within
            the heading angle bounds.

            Starts at the reference distance sRef and moves forward and
            backward, stopping once the heading angle exceeds the bounds.


            Returns:
                Indexes within the heading angle bounds. Unsorted and may
                contain duplicate indexes.
            """
            # Interpolate the index at the reference distance
            nCoords = len(self.sCoords)
            startInd = np.interp(sRef, self.sCoords, np.arange(nCoords))

            # Get arrays of all the indexes forward and backward of the reference distance
            indsForward = np.arange(np.ceil(startInd), nCoords, 1)
            indsBackward = np.arange(0, np.floor(startInd), 1)

            # If the track is closed, include the indexes wrapping around back to the start index (i.e. contain all indexes)
            # In all cases, flip the backwards index array so that incrementing the index equates to moving backwards around the track
            if BClosedTrack:
                indsForward = np.append(indsForward, indsBackward)
                indsBackward = np.flip(indsForward)
            else:
                indsForward = indsForward
                indsBackward = np.flip(indsBackward)

            # Find the indexes within the heading angle bounds in the forwards and backwards direction
            # In each direction, stop once the filtered heading angle exceeds the bounds
            inds_A = []
            for inds in (indsBackward, indsForward):
                for i in inds:
                    if ALower <= self.AHeadingsFilt[i] <= AUpper:
                        inds_A.append(i)
                    else:
                        break

            return np.array(inds_A)

        # Get the indexes of the coordinate array within the distance bounds, and within the heading angle bounds, separately

        # Get the indexes satisfying the distance or heading bounds, separately
        indsDistance = __getIndsDistance()
        indsHeading = __getIndsHeading()

        # Get the indexes satisfying both the distance and heading bounds, in ascending index order, then remove duplicates
        indsValid = np.union1d(indsDistance, indsHeading)
        indsValid = utils.removeConsecutiveDuplicates(indsValid)

        # If the track is closed, roll the valid indexes array such that the indexes are continuously incrementing by 1, except for the wrap back to 0
        # Technically only necessary if the valid indexes wrapped around the track
        if BClosedTrack:
            dindsValid = np.diff(indsValid)
            if np.max(dindsValid) > 1:
                offset = np.where(dindsValid > 1)[0][0] + 1
                indsValid = np.roll(indsDistance, -offset)

        def __getCoordOnBound(BStart: bool) -> tuple[NDArrayFloat1D, float, float, float]:
            """
            Calculates the coordinate on the bound limiting the start or finish
            of the reduced coordinate array.

            If the track is closed or the bound lies within the coordinate
            array, uses linear interpolation. Otherwise, if the bound lies
            beyond the defined coordinate array, extrapolates using the first/
            last low-pass filtered heading angle.

            Note that by definition, a coordinate is interpolated if it is on a
            heading bound.

            Args:
                BStart: Whether the coordinate on the bound to calculate is at
                    the start or finish of the coordinate array.

            Returns:
                Tuple of (xyzBound, sBound, ABound, AFiltBound).

                xyzBound: Coordinate at the bound, in the form [x, y, z].

                sBound: Cumulative distance at the bound (calculated along the
                    original coordinate array).

                ABound: Unwrapped heading angle at the bound (calculated along
                    the original coordinate array).

                AFiltBound: Low-pass filtered unwrapped heading angle at the
                    bound (calculated along the original coordinate array).
            """
            # Get the index at the start/finish of the valid indexes satisfying both the distance and heading angle bounds to calculate the coordinate
            # on the bound
            indsIndex = 0 if BStart else -1

            if indsValid[indsIndex] in indsHeading:
                # On a distance bound (lower bound if indsIndex is 0, upper bound if indsIndex is -1), wrap the bound if the track is closed
                sBound = sLower if indsIndex == 0 else sUpper
                if BClosedTrack:
                    sBound = utils.wrap(sBound, 0, self.sCoords[-1])

                # Check whether to interpolate or extrapolate
                if BClosedTrack or 0 <= sBound <= self.sCoords[-1]:
                    # Closed track or distance bound within the coordinate array, use linear interpolation
                    xyzBound = np.array([np.interp(sBound, self.sCoords, self.xyzCoords[:, 0]),
                                         np.interp(sBound, self.sCoords, self.xyzCoords[:, 1]),
                                         np.interp(sBound, self.sCoords, self.xyzCoords[:, 2])])
                    ABound = np.interp(sBound, self.sCoords, self.AHeadings)
                    AFiltBound = np.interp(sBound, self.sCoords, self.AHeadingsFilt)
                else:
                    # Distance bound beyond the coordinate array, get the index of the closest distance to the distance bound
                    indLimit = 0 if sBound < 0 else -1

                    # Get the low-pass filtered unwrapped heading angle at the closest distance to the distance bound, and also assume this angle for
                    # the raw unwrapped heading angle at this coordinate
                    AFiltBound = self.AHeadingsFilt[indLimit]
                    ABound = AFiltBound

                    # Extrapolate the coordinate at this angle
                    xyzVec = utils.rotateVectorHeading(np.array([0, 1, 0]) * (sBound - self.sCoords[indLimit]), AFiltBound)
                    xyzBound = self.xyzCoords[indLimit] + xyzVec

            else:
                # On a heading angle bound, find which of the heading bounds is limiting (which of the heading bounds is closer)
                AFiltBound = ALower if self.AHeadingsFilt[indsHeading[indsIndex]] < (ALower + AUpper) / 2 else AUpper

                # Get the indexes surrounding this heading bound
                if indsIndex == 0:
                    indLower = indsHeading[indsIndex] - 1
                    indUpper = indsHeading[indsIndex]
                else:
                    indLower = indsHeading[indsIndex]
                    indUpper = indsHeading[indsIndex] + 1

                # Get the coordinate array attributes at the indexes surrounding this heading bound, ordered with increasing filtered heading angle
                if self.AHeadingsFilt[indLower] < self.AHeadingsFilt[indUpper]:
                    i1 = indLower
                    i2 = indUpper
                else:
                    i1 = indUpper
                    i2 = indLower
                xyz = np.array([self.xyzCoords[i1], self.xyzCoords[i2]])
                s = np.array([self.sCoords[i1], self.sCoords[i2]])
                A = np.array([self.AHeadings[i1], self.AHeadings[i2]])
                AFilt = np.array([self.AHeadingsFilt[i1], self.AHeadingsFilt[i2]])

                # Linearly interpolate the coordinate array attributes at the heading bound - note that utils.linearInterpExtrap() is used as it is
                # faster when the function is defined only by 2 coordinates (which it has to be as AHeadingsFilt is not guaranteed to be monotonically
                # increasing)
                xyzBound = np.array([utils.linearInterpExtrap(AFiltBound, AFilt, xyz[:, 0]),
                                     utils.linearInterpExtrap(AFiltBound, AFilt, xyz[:, 1]),
                                     utils.linearInterpExtrap(AFiltBound, AFilt, xyz[:, 2])])
                sBound = utils.linearInterpExtrap(AFiltBound, AFilt, s)
                ABound = utils.linearInterpExtrap(AFiltBound, AFilt, A)

            return xyzBound, sBound, ABound, AFiltBound

        # Calculate the coordinates and their attributes at the start and finish of the reduced coordinate array (i.e. the coordinates on the bounds)
        xyzStart, sStart, AStart, AFiltStart = __getCoordOnBound(True)
        xyzFinish, sFinish, AFinish, AFiltFinish = __getCoordOnBound(False)

        # Combine the calculated values on the bounds with the values of the coordinate array from the valid indexes
        xyzCoords = np.vstack((xyzStart, self.xyzCoords[indsValid], xyzFinish))
        sCoords = np.vstack((sStart, self.sCoords[indsValid], sFinish))
        AHeadings = np.vstack((AStart, self.AHeadings[indsValid], AFinish))
        AFiltHeadings = np.vstack((AFiltStart, self.AHeadingsFilt[indsValid], AFiltFinish))

        # Return a new CoordinateArray instance initialised with its attributes provided
        # Note that the BAllowNegativeInitAHeading flag has no effect when initialised with attributes provided as is the case here
        # Also note that any consecutive duplicates will be removed during the initialisation of the new CoordinateArray instance
        return CoordinateArray(xyzCoords, False, False, sCoords, AHeadings, AFiltHeadings)


@dataclass
class Event:

    """
    Event data, which is then stored in the corresponding Gate object.

    Custom event types:
        - TODO: Implement the custom event types (DRS, SLM, SpeedLimiter etc.)

    Internal event types:
        - StartFinish
        - GateCreation

    Attributes:
        name: Name of the event gates pair. Note that for each event gate name,
            there must be a start and finish gate.
        type: Event type, see above for the valid event gate types.
        BStart: Whether the gate object storing the event marks the start or
            finish of the event.
        properties: Dictionary containing the properties specific to the event
            type required to completely define it. Empty if the event type does
            not require any additional information to be defined.
    """
    name: str
    type: CUSTOM_EVENT_TYPES | INTERNAL_EVENT_TYPES
    BStart: bool
    properties: dict[str, Any]


class Gate:
    """
    Information about the gate.

    The Gate object is represented only on the 2D plane [x, y] - otherwise
    handling intersections with track limits would be practically impossible.
    Gates are used to define event start and finish locations and define the
    track limits used to validate generated trajectories.

    Has methods:
        - calcIntersection(reducedCoordArray, key)
        - updateWidths(lLeft, lRight)
        - recalcMidpoint()

    Attributes:
        line: Straight line from the left coordinate to the right coordinate, in
            the 2D plane [x, y]. Shapely LineString object.
        xyMidpoint: Midpoint between the left and right soft track limits,
            measured along the line of the gate in the 2D plane [x, y]. 1D array
            of floats.
        AHeading: Unwrapped heading angle of the gate, in radians. 0 corresponds
            to positive y (north) and increasing clockwise. This should align
            with the unwrapped heading angles of the CoordinateArray objects.
            Float.
        event: Information about the event starting/ending at this gate. If this
            is None, then this is a "regular" gate and does not define an event
            start or finish location. Event object or None.
        lLeft: Width of the gate to the left of xyMidpoint. Float.
        lRight: Width of the gate to the right of xyMidpoint. Float.
        lLimitLeftSoft: Unsigned distance from xyMidpoint to the intersection
            with the left soft track limit. Float or None.
        lLimitRightSoft: Unsigned distance from xyMidpoint to the intersection
            with the right soft track limit. Float or None.
        lLimitLeftHard: Unsigned distance from xyMidpoint to the intersection
            with the left hard track limit. Float or None.
        lLimitRightHard: Unsigned distance from xyMidpoint to the intersection
            with the right hard track limit. Float or None.
    """
    def __init__(self,
                 xyMidpoint: NDArrayFloat1D,
                 AHeading: float,
                 event: Event | None = None,
                 lLeft: float = GATE_MAX_WIDTH / 2,
                 lRight: float = GATE_MAX_WIDTH / 2,
                 lLimitLeftSoft: float | None = None,
                 lLimitRightSoft: float | None = None,
                 lLimitLeftHard: float | None = None,
                 lLimitRightHard: float | None = None) -> None:
        """
        Initialises the Gate object, setting all the attributes provided and
        generating the gate line.

        Args:
            xyMidpoint: Midpoint of the gate between the left and right soft
                track limits, in the 2D plane [x, y].
            AHeading: Unwrapped heading angle of the gate, in radians. 0
                corresponds to positive y (north) and increasing clockwise. This
                should align with the unwrapped heading angles of the
                CoordinateArray objects.
            event: Information about the event starting/ending at this gate. If
                this is None, then this is a "regular" gate and does not define
                an event start or finish location.
            lLeft: Width of the gate to the left of xyMidpoint.
            lRight: Width of the gate to the right of xyMidpoint.
            lLimitLeftSoft: Unsigned distance from xyMidpoint to the
                intersection with the left soft track limit.
            lLimitRightSoft: Unsigned distance from xyMidpoint to the
                intersection with the right soft track limit.
            lLimitLeftHard: Unsigned distance from xyMidpoint to the
                intersection with the left hard track limit.
            lLimitRightHard: Unsigned distance from xyMidpoint to the
                intersection with the right hard track limit.
        """
        # Set raw attributes
        self.xyMidpoint = xyMidpoint
        self.AHeading = AHeading
        self.event = event
        self.lLeft = lLeft
        self.lRight = lRight
        self.lLimitLeftSoft = lLimitLeftSoft
        self.lLimitRightSoft = lLimitRightSoft
        self.lLimitLeftHard = lLimitLeftHard
        self.lLimitRightHard = lLimitRightHard

        # Create the gate line
        self.line = self.__createGateLine()

    def __createGateLine(self) -> shapely.LineString:
        """
        Internal method to create the gate Shapely LineString object from the
        gate attributes.

        Returns:
            Shapely LineString representing the gate, which is a straight line
            from the left to the right coordinate.
        """
        # Calculate the vectors to the left and right coordinates of the gate, relative to xyMidpoint
        xyVecLeft = utils.rotateVectorHeading(np.array([-self.lLeft, 0]), self.AHeading)
        xyVecRight = utils.rotateVectorHeading(np.array([self.lRight, 0]), self.AHeading)

        # Calculate the left and right coordinates of the gate
        xyLeft = self.xyMidpoint + xyVecLeft
        xyRight = self.xyMidpoint + xyVecRight

        # Generate the gate line
        return shapely.LineString([xyLeft, xyRight])

    def calcIntersection(self,
                         reducedCoordArray: CoordinateArray,
                         key: str = '') -> tuple[float, float]:
        """
        Calculate the gate intersection with the reduced CoordinateArray object
        provided.

        Finds the distance from the gate xyMidpoint to the intersection point,
        and the distance along the (original) CoordinateArray object of the
        intersection.

        Updates the relevant lLimitXY attribute for the key provided.

        Args:
            reducedCoordArray: Reduced CoordinateArray object for the local
                region around the gate.
            key: Name of the coordinate array passed in. Automatically updates
                the relevant lLimitXY attribute if the key is one of
                'limitLeftSoft', 'limitRightSoft', 'limitLeftHard' or
                'limitRightHard'.

        Returns:
            Tuple of (lIntersection, sIntersection).

            lIntersection: Unsigned distance from the gate midpoint to the
                intersection between the gate and reducedCoordArray. If there
                is no intersection, this will be GATE_MAX_WIDTH / 2.

            sIntersection: Distance along reducedCoordArray of the intersection
                with the gate, as calculated from its sCoords attribute. If
                there is no intersection, this will be the distance along
                reducedCoordArray of the closest point to the gate midpoint.
        """
        # Create a Shapely LineString from the reduced coordinate array
        lineCoordArray = shapely.LineString(reducedCoordArray.xyzCoords[:2])

        # Find the geometry shared between the gate and the reduced coordinate array - note that as the gate is a straight line and lineCoordArray is
        # a line, their shared geometry can only be Shapely Point or Shapely LineString objects, where the LineString objects must be straight lines
        # defined by their start and finish coordinates (unless there is no intersection
        intersections = shapely.intersection(self.line, lineCoordArray)

        # Check if there was an intersection
        if intersections.is_empty:
            # No intersection between the gate and the reduced coordinate array, set lIntersection to half of the gate max width, and sIntersection to
            # the distance along the coordinate array of the closest point to the gate midpoint
            lIntersection = GATE_MAX_WIDTH / 2
            sIntersection = reducedCoordArray.sCoords[0] + lineCoordArray.project(shapely.Point(self.xyMidpoint))

        else:
            # Gate intersects with the reduced coordinate array, get a list of the intersection geometries
            intersectionsList = intersections.geoms if isinstance(intersections, shapely.GeometryCollection) else [intersections]

            # Find the lowest intersection distance and its corresponding [x, y] coordinate
            lIntersection = GATE_MAX_WIDTH / 2
            xyIntersection = self.xyMidpoint
            for intersection in intersectionsList:
                if isinstance(intersection, shapely.Point):
                    # This intersection is a point, get its coordinates as an array, then calculate the distance to the intersection
                    xy = np.array([intersection.x, intersection.y])
                    l = np.linalg.norm(self.xyMidpoint - xy)

                else:
                    # This intersection is a line, get the coordinates defining the intersection line as a 2D array in the form [[x1, y1], [x2, y2]]
                    xyLine = np.array([intersection.xy[0], intersection.xy[1]]).T

                    # Check if the gate midpoint is contained in the intersection line
                    if xyLine[0][0] <= self.xyMidpoint[0] <= xyLine[1][0] and xyLine[0][1] <= self.xyMidpoint[1] <= xyLine[1][1]:
                        # Gate midpoint contained within the intersection line, meaning the reduced coordinate array passes through the gate midpoint
                        xy = self.xyMidpoint
                        l = 0
                    else:
                        # Gate midpoint not contained within the intersection line, find the shorter distance to the intersection line and its
                        # coordinate
                        lIntersections = np.linalg.norm(self.xyMidpoint - xyLine, ord=1, axis=1)
                        if lIntersections[0] < lIntersections[1]:
                            xy = xyLine[0]
                            l = lIntersections[0]
                        else:
                            xy = xyLine[1]
                            l = lIntersections[1]

                # Update the lowest intersection distance and its corresponding [x, y] coordinate
                if l < lIntersection:
                    lIntersection = l
                    xyIntersection = xy

            # Find the distance along the coordinate array of the closest intersection
            sIntersection = reducedCoordArray.sCoords[0] + lineCoordArray.project(shapely.Point(xyIntersection))

        # Update the relevant lLimitXY attribute
        if key == 'limitLeftSoft':
            self.lLimitLeftSoft = lIntersection
        elif key == 'limitRightSoft':
            self.lLimitRightSoft = lIntersection
        elif key == 'limitLeftHard':
            self.lLimitLeftHard = lIntersection
        elif key == 'limitRightHard':
            self.lLimitRightHard = lIntersection

        return lIntersection, sIntersection


    def updateWidths(self,
                     lLeft: float,
                     lRight: float) -> None:
        """
        Updates the gate width attributes, then updates the gate line from those
        new width attributes.

        Args:
            lLeft: Width of the gate to the left of xyMidpoint.
            lRight: Width of the gate to the right of xyMidpoint.
        """
        # Update attributes
        self.lLeft = lLeft
        self.lRight = lRight

        # Create the new gate line using the updated attributes
        self.line = self.__createGateLine()

    def recalcMidpoint(self) -> None:
        """
        Recalculates and moves the gate midpoint to the true midpoint between
        the left and right soft track limits.

        The resulting distances to the left and right soft track limits will be
        equal. Also updates all the gate attributes with the new values (if they
        exist).

        Raises:
            ValueError: Cannot recalculate gate midpoint: lLimitLeftSoft and/or
                lLimitRightSoft is None
        """
        # Check that lLimitLeftSoft and lLimitRightSoft both exist
        if self.lLimitLeftSoft is None or self.lLimitRightSoft is None:
            raise ValueError("Cannot recalculate gate midpoint: lLimitLeftSoft and/or lLimitRightSoft is None")

        # Calculate the distance to the right to shift the midpoint
        lShiftRight = self.lLimitRightSoft - self.lLimitLeftSoft

        # Calculate the shifted gate midpoint and update the attribute
        xyVec = utils.rotateVectorHeading(np.array([lShiftRight, 0]), self.AHeading)
        self.xyMidpoint += xyVec

        # Update the width and distance to limit attributes with the shifted midpoint
        self.lLeft += lShiftRight
        self.lRight -= lShiftRight
        self.lLimitLeftSoft += lShiftRight
        self.lLimitRightSoft -= lShiftRight
        if self.lLimitLeftHard is not None:
            self.lLimitLeftHard += lShiftRight
        if self.lLimitRightHard is not None:
            self.lLimitRightHard -= lShiftRight

        # Create the new gate line using the updated attributes
        self.line = self.__createGateLine()


class Track:
    """
    Defines the track on which a trajectory can be created and optimised.

    Has the functions calcTrackZ() and calcTrackNormal() to calculate the z
    coordinate and [x, y, z] unit normal vector at a given [x, y] coordinate on
    the track.
    TODO: More detailed docstring
    """
    def __init__(self,
                 trackPath: str,
                 BClosedTrackOverride: bool | None = None,
                 BForceTrackGen: bool = False) -> None:
        """
        TODO: Function docstring
        """
        ...

    def __initFromPkl(self,
                      pklPath: str) -> None:
        """
        TODO: Function docstring
        """
        ...

    def __initFromTrackGen(self,
                           trackPath: str,
                           BClosedTrackOverride: bool | None) -> None:
        """
        TODO: Docstring
        """
        ...
        def __initGateFromCoords(xyLeft: NDArrayFloat1D,
                                 xyRight: NDArrayFloat1D,
                                 BAllowNegativeAHeading: bool) -> Gate:
            """
            TODO: Function docstring
            """
            ...

        def __parseTrackFiles(trackPath: str) -> tuple[dict[str, CoordinateArray], dict[str, Gate]]:
            """
            TODO: Function docstring
            """
            ...

        def __saveTrackPlot(trackPath: str,
                            coordArraysDict: dict[str, Any],
                            gates: list[Gate] | np.ndarray[tuple[int], np.dtype[Gate]],
                            badGateInds: list[int] | None = None) -> None:
            """
            TODO: Function docstring
            """
            ...
        ...

    def calcTrackZ(self,
                   xy: list[float] | NDArrayFloat1D,
                   gateInd: int,
                   BReturnNaN: bool = False) -> float:
        """
        TODO: Function docstring
        """
        ...

    def calcTrackNormal(self,
                        xy_xyz: list[float] | NDArrayFloat1D,
                        gateInd: int) -> NDArrayFloat1D:
        """
        TODO: Function docstring
        """
        ...

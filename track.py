"""
The track module is responsible for defining the track on which a trajectory
can be created and optimised.

This includes the Track class, as well as the CoordinateArray, Event and Gate
classes used to generate and define the track.
"""
# Python standard libraries
import os
import csv
import json
import pickle as pkl
from typing import Literal
from dataclasses import dataclass

# External libraries
import scipy
import shapely
import numpy as np

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
                                                # allow suitable penalties to be calculated if track limits are violated (m)
                                                #  - Higher gives more robustness, allowing a trajectory that exceeds track limits to still be solved
                                                #  - Too high can cause gates to be skipped in tight corners due to overlapping, losing track limits
                                                #    resolution

# Track generation constants
CLOSED_TRACK_THRESHOLD_DISTANCE = 20            # Maximum distance (as the crow flies) from the start to finish coordinates of the provided soft track
                                                # limits to consider the track closed, when using the automatic logic (m)

HEADING_ANGLE_THRESHOLD = np.pi / 4             # Threshold for the difference in heading angles (where the difference is wrapped from -pi to pi then
                                                # the absolute value is taken), above which to consider the heading angles as "similar"

GATE_STEP_DISTANCE = 5                          # Distance between each consecutive gate for gate creation, unless overridden by an event gate or the
                                                # gate is skipped as it overlaps with neighbouring gates (m)
                                                #  - Higher gives more resolution for determining track limits
                                                #  - Too high may excessively slow track and trajectory generation

REDUCED_DISTANCE_WINDOW = 250                   # Window on each side for the distance bounds of the reduced coordinate array for gate creation (m)

REDUCED_HEADING_WINDOW = np.pi                  # Window on each side for the heading angle bounds of the reduced coordinate array for gate creation
                                                # (radians)

# Track plot constants
TRACK_PLOT_AREA = 100                           # Area of the track plot saved after track generation or if track generation raised an exception
                                                # manually (square inches)

TRACK_PLOT_SPATIAL_RESOLUTION = 5               # Spatial resolution of the saved track plot (in pixels/m)

# Track normal vector constants
PERTURB_DISTANCE = 1e-3                         # Distance to perturb when forward-differencing to calculate the track normal vector


class CoordinateArray:
    """
    Array of coordinates and its derived attributes.

    Used for track generation to store the coordinates defining the track limits
    and track mesh.

    Has methods:
        - rotateHeading(AHeading)
        - getReducedCoordArray(sRef, sLower, sUpper, ALower, AUpper)

    Attributes:
        xyzCoords: Coordinates in order of increasing distance along the
            coordinate array, where each coordinate is in the form [x, y, z].
            2D array of floats.
        xyLine: Line formed by the coordinates in the 2D plane [x, y]. Shapely
            LineString.
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

        Note that the initial unfiltered heading angle will be between 0 and 2
        pi. This should be checked and corrected if necessary with the method
        rotateHeadings().

        Args:
            xyzCoords: Array of coordinates in order of increasing distance
                along the track, where each coordinate is in the form [x, y, z].
            BClosedTrack: Whether the coordinate array should be treated as
                closed. A closed coordinate array means that the last coordinate
                is equal to the first coordinate.
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
        # Make sure this is a closed coordinate array if the track is closed (last coordinate equal to the first coordinate)
        if BClosedTrack and xyzCoords[-1] != xyzCoords[0]:
            xyzCoords = np.vstack((xyzCoords, xyzCoords[0]))
        else:
            xyzCoords = utils.removeConsecutiveDuplicates(xyzCoords, axis=0)

        # Get the indexes that omit consecutive duplicates from the coordinate array, and remove the consecutive duplicates from the coordinate array,
        # then set the xyzCoords attribute
        inds = utils.getIndsWithoutConsecutiveDuplicates(xyzCoords, axis=0)
        self.xyzCoords = xyzCoords[inds]

        # Generate the Shapely LineString of the coordinates in the 2D plane [x, y]
        self.xyLine = shapely.LineString(self.xyzCoords[:2])

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

    def rotateHeadings(self,
                       theta: float) -> None:
        """
        Offsets the attributes AHeadings and AHeadingsFilt by theta.

        Should be used to correct the calculated heading angle attributes to
        ensure that all CoordinateArray objects are in the same rotation.

        Args:
            theta: Heading angle in radians to offset the attributes.
        """
        self.AHeadings += theta
        self.AHeadingsFilt += theta

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
        xyLine: Straight line from the left coordinate to the right coordinate, in
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
        self.xyLine = self.__getGateLine()

    def __getGateLine(self) -> shapely.LineString:
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
        intersections = shapely.intersection(self.xyLine, lineCoordArray)

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
                     lLeft: float | None = None,
                     lRight: float | None = None) -> None:
        """
        Updates the gate width attributes, then updates the gate line from those
        new width attributes.

        Args:
            lLeft: Width of the gate to the left of xyMidpoint. If not provided,
                uses lLimitLeftHard + GATE_EXTEND_WIDTH.
            lRight: Width of the gate to the right of xyMidpoint. If not
                provided, uses lLimitRightHard + GATE_EXTEND_WIDTH.

        Raises:
            ValueError 1: Cannot update gate widths: lLeft not specified but
                default value cannot be used as lLimitLeftHard is None

            ValueError 2: Cannot update gate widths: lRight not specified but
                default value cannot be used as lLimitRightHard is None
        """
        # Check if lLeft and lRight were not provided and set their default values
        if lLeft is None:
            if self.lLimitLeftHard is None:
                raise ValueError("Cannot update gate widths: lLeft not specified but default value cannot be used as lLimitLeftHard is None")
            lLeft = self.lLimitLeftHard + GATE_EXTEND_WIDTH
        if lRight is None:
            if self.lLimitRightHard is None:
                raise ValueError("Cannot update gate widths: lRight not specified but default value cannot be used as lLimitRightHard is None")
            lRight = self.lLimitRightHard + GATE_EXTEND_WIDTH

        # Update attributes
        self.lLeft = lLeft
        self.lRight = lRight

        # Create the new gate line using the updated attributes
        self.xyLine = self.__getGateLine()

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
        self.xyLine = self.__getGateLine()


class Track:
    """
    Defines the track on which a trajectory can be created and optimised.

    Has the functions calcTrackZ() and calcTrackNormal() to calculate the z
    coordinate and [x, y, z] unit normal vector at a given [x, y] coordinate on
    the track.

    Has methods:
        - calcTrackZ(xy, indGate, BReturnNaN)
        - calcTrackNormal(xy_xyz, indGate)

    Attributes:
        gates: Track gates, with order corresponding to the forwards direction
            around the track. NumPy array of Gate objects.
        mesh: Interpolators for the local track z coordinate around each gate.
            NumPy array of SciPy multivariate interpolators.
        BClosedTrack: Whether the track forms a closed circuit.
    """
    def __init__(self,
                 trackPath: str,
                 BClosedTrackOverride: bool | None = None,
                 BForceTrackGen: bool = False) -> None:
        """
        Initialises the Track object, by loading the Track object from a .pkl
        file in the specified folder, or if that fails, by generating the track
        from the .csv and .json files in that folder.

        TODO: Required file structure in trackPath

        Args:
            trackPath: File path to the folder containing the .csv and .json
                files required for track generation.
            BClosedTrackOverride: Overrides the automatic logic for determining
                if the track is a closed circuit. If this is None or not
                provided, whether the track is closed or not is determined by
                the maximum distance from the start to finish coordinates of the
                track limit coordinate arrays. This has no effect if the track
                is loaded from a .pkl file.
            BForceTrackGen: If true, overrides and forces the track to be
                generated and will overwrite the track .pkl file in the
                specified folder. If false, will use the automatic logic of
                loading from the .pkl file, and falling back to generating the
                track if it fails.
        """
        if not BForceTrackGen:
            try:
                # Try to load and initialise from the .pkl file
                self.__initFromPkl(trackPath)
            except Exception:
                # TODO: Find out what exceptions can happen to be more specific
                # Initialisation from .pkl failed
                BForceTrackGen = True

        if BForceTrackGen:
            # Fall back to initialisation from track generation
            self.__initFromTrackGen(trackPath, BClosedTrackOverride)


    def __initFromPkl(self,
                      trackPath: str) -> None:
        """
        Internal method to initialise the Track object from a .pkl file.

        Args:
            trackPath: File path to the folder containing the track .pkl file.
        """
        # Load the .pkl file
        with open(os.path.join(trackPath, TRACK_PKL_FILENAME), 'rb') as pklFile:
            trackPkl = pkl.load(pklFile)

        # Set all attributes from the attributes in the .pkl file
        self.gates = trackPkl.gates
        self.mesh = trackPkl.mesh
        self.BClosedTrack = trackPkl.BClosedTrack

    def __initFromTrackGen(self,
                           trackPath: str,
                           BClosedTrackOverride: bool | None) -> None:
        """
        Internal method to initialise the Track object by generating the track
        from the .csv and .json files in the specified path.

        TODO: Calculation logic

        Args:
            trackPath: File path to the folder containing the .csv and .json
                files required for track generation.
            BClosedTrackOverride: Overrides the automatic logic for determining
                if the track is a closed circuit. If this is None or not
                provided, whether the track is closed or not is determined by
                the maximum distance from the start to finish coordinates of the
                track limit coordinate arrays.
        """
        def __getGateFromCoords(xyLeft: NDArrayFloat1D,
                                xyRight: NDArrayFloat1D,
                                event: Event | None = None,
                                lLimitLeftSoft: float | None = None,
                                lLimitRightSoft: float | None = None,
                                lLimitLeftHard: float | None = None,
                                lLimitRightHard: float | None = None) -> Gate:
            """
            Initialise a Gate object from its left and right coordinates.

            Note that the heading angle generated may be in the wrong rotation
            cycle from the desired heading angle. This attribute should be
            checked and modified after the gate is returned.

            Args:
                xyLeft: Left coordinate of the gate, in the form [x, y].
                xyRight: Right coordinate of the gate, in the form [x, y].
                event: Information about the event starting/ending at this gate.
                    If this is None, then this is a "regular" gate and does not
                    define an event start or finish location.
                lLimitLeftSoft: Unsigned distance from xyMidpoint to the
                    intersection with the left soft track limit.
                lLimitRightSoft: Unsigned distance from xyMidpoint to the
                    intersection with the right soft track limit.
                lLimitLeftHard: Unsigned distance from xyMidpoint to the
                    intersection with the left hard track limit.
                lLimitRightHard: Unsigned distance from xyMidpoint to the
                    intersection with the right hard track limit.

            Returns:
                Gate object.
            """
            # Calculate gate midpoint, heading angle and widths to the left and right coordinates from the midpoint
            xyMidpoint = (xyLeft + xyRight) / 2
            AHeading = utils.getHeading(xyRight - xyLeft) - (np.pi / 4)
            lHalf = float(np.linalg.norm(xyRight - xyLeft) / 2)

            # Initialise gate
            return Gate(xyMidpoint, AHeading, event, lHalf, lHalf, lLimitLeftSoft, lLimitRightSoft, lLimitLeftHard, lLimitRightHard)

        def __parseTrackFiles() -> tuple[dict[str, CoordinateArray], dict[str, list[Gate]]]:
            """
            TODO: Function docstring
            """
            limitFileNames = ('xyzLimitLeftSoft.csv', 'xyzLimitRightSoft.csv', 'xyzLimitLeftHard.csv', 'xyzLimitRightHard.csv')
            extraCoordsFileNamePrefix = 'xyzExtra'
            eventGatesFileNamePrefix = 'eventData'
            validEventTypes = list(CUSTOM_EVENT_TYPES) + ['StartFinish']

            # Parse all provided track files
            xyzCoordsDict = {}
            eventGatesDict: dict[str, list[Gate]] = {}
            with os.scandir(trackPath) as entries:
                for entry in entries:
                    if entry.name in limitFileNames or (entry.name.startswith(extraCoordsFileNamePrefix) and entry.name.endswith('.csv')):
                        # Coordinate array .csv file, read the coordinates to the relevant key in xyzCoordArraysDict
                        key = entry.name[3:-4] if entry.name in limitFileNames else entry.name[len(extraCoordsFileNamePrefix) - 1:-4]
                        with open(entry, newline='') as csvFile:
                            xyzCoordsDict[key] = np.array(csv.reader(csvFile)).astype(float)

                    elif entry.name.startswith(eventGatesFileNamePrefix) and entry.name.endswith('json'):
                        # Event data .json file, read it and initialise the event gates to append to the relevant lists in eventGatesDict
                        with open(entry, 'r') as jsonFile:
                            eventData = json.load(jsonFile)
                        dataKeys = eventData.keys()

                        # Parse the event type and event name, using the name 'StartFinish' if the gate type is 'StartFinish', otherwise parsing it
                        # from the file name
                        if 'type' in dataKeys:
                            eventType = eventData['type']
                            if eventType not in validEventTypes:
                                raise ValueError(f"Event data file {entry.name} has an invalid event type {eventType}: valid event types are "
                                                 f"{validEventTypes}")
                            eventName = 'StartFinish' if eventType == 'StartFinish' else entry.name[len(eventGatesFileNamePrefix) - 1:-5]
                        else:
                            raise ValueError(f"Event data file {entry.name} does not have the key 'type' required")

                        # Parse the coordinates of the event gate representing the start and finish of the event
                        for subKey in ('Start', 'Finish'):
                            if f'xy{subKey}Left' in dataKeys and f'xy{subKey}Right' in dataKeys:
                                event = Event(eventName, eventType, subKey == 'Start', eventData.get('properties', {}))
                                eventGate = __getGateFromCoords(eventData[f'xy{subKey}Left'], eventData[f'xy{subKey}Right'], event)
                                if eventType in eventGatesDict.keys():
                                    eventGatesDict[eventType].append(eventGate)
                                else:
                                    eventGatesDict[eventType] = [eventGate]
                            elif eventType != 'StartFinish':
                                raise ValueError(f"Event data file {entry.name} does not have the {subKey.lower()} coordinate keys 'xyStartLeft' "
                                                 f"and/or 'xyStartRight', and event type is '{eventType}', not 'StartFinish'")

            # Use fallbacks for left and right track limits coordinates if they don't exist, and automatically determine if the track is closed
            BClosedTrack = True
            AHeadingsFiltInitAvg = 0
            xyzCoordsKeys = xyzCoordsDict.keys()
            for subKey in ('Left', 'Right'):
                # Check if fallbacks are needed
                if f'limit{subKey}Soft' not in xyzCoordsKeys or f'limit{subKey}Hard' not in xyzCoordsKeys:
                    # Fallbacks needed, use them
                    if f'limit{subKey}Soft' in xyzCoordsKeys and f'limit{subKey}Hard' not in xyzCoordsKeys:
                        xyzCoordsDict[f'limit{subKey}Hard'] = xyzCoordsDict[f'limit{subKey}Soft']
                    elif f'limit{subKey}Soft' not in xyzCoordsKeys and f'limit{subKey}Hard' in xyzCoordsKeys:
                        xyzCoordsDict[f'limit{subKey}Soft'] = xyzCoordsDict[f'limit{subKey}Hard']
                    else:
                        raise ValueError(f"{trackPath} must contain at least one of xyzLimit{subKey}Soft.csv or xyzLimit{subKey}Hard.csv")

                # Determine if the track is closed, if all track limits coordinates have their start and finish coordinates within
                # CLOSED_TRACK_THRESHOLD_DISTANCE
                for subKey2 in ('Soft', 'Hard'):
                    xyzCoords = xyzCoordsDict[f'limit{subKey}{subKey2}']
                    if np.linalg.norm(xyzCoords[-1] - xyzCoords[0]) > CLOSED_TRACK_THRESHOLD_DISTANCE:
                        BClosedTrack = False

            # Set BClosedTrack attribute
            self.BClosedTrack = BClosedTrack if BClosedTrackOverride is None else BClosedTrackOverride

            # Initialise CoordinateArray objects and store them in the relevant key of coordArraysDict, also calculate the average initial filtered
            # heading angle
            AHeadingsFiltInitAvg = 0
            coordArraysDict: dict[str, CoordinateArray] = {}
            for key, xyzCoords in xyzCoordsDict.items():
                coordArray = CoordinateArray(xyzCoords, self.BClosedTrack)
                coordArraysDict[key] = coordArray
                AHeadingsFiltInitAvg += coordArray.AHeadingsFilt[0]
            AHeadingsFiltInitAvg /= len(xyzCoordsDict)

            # Make sure all the coordinate arrays are in the same rotation
            ALower = AHeadingsFiltInitAvg - np.pi
            AUpper = AHeadingsFiltInitAvg + np.pi
            for coordArray in coordArraysDict.values():
                AOffset = utils.wrap(coordArray.AHeadingsFilt[0], ALower, AUpper) - coordArray.AHeadingsFilt[0]
                coordArray.rotateHeadings(AOffset)

            # Check if start and finish gates exist
            startGate = None
            finishGate = None
            if 'StartFinish' in eventGatesDict.keys():
                for eventGate in eventGatesDict['StartFinish']:
                    if eventGate.event.BStart:
                        startGate = eventGate
                    else:
                        finishGate = eventGate

            # Create start and finish gates if they don't exist
            if startGate is None:
                startGate = __getGateFromCoords(coordArraysDict['limitLeftSoft'].xyzCoords[0],
                                                coordArraysDict['limitRightSoft'].xyzCoords[0],
                                                Event('StartFinish', 'StartFinish', True, {}))
            if finishGate is None:
                finishGate = __getGateFromCoords(coordArraysDict['limitLeftSoft'].xyzCoords[-1],
                                                 coordArraysDict['limitRightSoft'].xyzCoords[-1],
                                                 Event('StartFinish', 'StartFinish', False, {}))

            # Make sure the 'StartFinish' key in eventGatesDict is in the right order (rewrite its value)
            eventGatesDict['StartFinish'] = [startGate, finishGate]

            # Make sure that every event type has an equal number of start and finish gates
            for eventGatesList in eventGatesDict.values():
                nStart = 0
                nFinish = 0
                for eventGate in eventGatesList:
                    if eventGate.event.BStart:
                        nStart += 1
                    else:
                        nFinish += 1
                if nStart != nFinish:
                    raise ValueError(f"Event type '{eventType}' has start gates but {nFinish} finish gates")

            return coordArraysDict, eventGatesDict

        def __getGateFromParams(params: NDArrayFloat1D,
                                prevGate: Gate) -> Gate:
            """
            Internal method to create a Gate object from the optimisation
            parameters.

            Args:
                params: Optimisation parameters in the form [psi, theta]. psi is
                    the clockwise angle in radians from the previous gate
                    heading to place the new gate. theta is the clockwise
                    heading angle offset in radians of the new gate, relative to
                    psi.
                prevGate: Previous gate, can be a normal track gate, skipped
                    gate, or event gate.

            Returns:
                Gate object created from the optimisation parameters.
            """
            # Unpack the params
            psi = params[0]     # Clockwise angle in radians from the previous gate heading to place the new gate
            theta = params[1]   # Clockwise heading angle offset in radians of the new gate, relative to psi

            # Calculate the vector from the previous gate's midpoint to the candidate gate midpoint
            xyVec = utils.rotateVectorHeading(np.array([0, GATE_STEP_DISTANCE]), prevGate.AHeading + psi)

            # Create the candidate gate
            candidateGate = Gate(prevGate.xyMidpoint + xyVec, prevGate.AHeading + psi + theta)

            return candidateGate

        def __gateObjFuncRoot(params: NDArrayFloat1D,
                              prevGate: Gate,
                              reducedLimitLeftSoft: CoordinateArray,
                              reducedLimitRightSoft: CoordinateArray) -> NDArrayFloat1D:
            """
            Internal method to get the objective function vector of the
            candidate gate, for the root-finding gate placement optimisation
            method.

            Args:
                params: Optimisation parameters in the form [psi, theta]. psi is
                    the clockwise angle in radians from the previous gate
                    heading to place the new gate. theta is the clockwise
                    heading angle offset in radians of the new gate, relative to
                    psi.
                prevGate: Previous gate, can be a normal track gate, skipped
                    gate, or event gate.
                reducedLimitLeftSoft: Reduced left soft track limit
                    CoordinateArray object.
                reducedLimitRightSoft: Reduced right soft track limit
                    CoordinateArray object.

            Returns:
                Objective function vector as an array where the first element is
                the signed difference in distances to the left and right soft
                track limits, and the second element is the signed difference in
                the angles that the soft track limits are widening/narrowing on
                the left and right sides.
            """
            # Get the candidate gate
            candidateGate = __getGateFromParams(params, prevGate)

            # Calculate the candidate gate's intersections with the soft track limits
            lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedLimitLeftSoft, 'limitLeftSoft')
            lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedLimitRightSoft, 'limitRightSoft')

            # Create the array representing the objective function
            objFunc = np.empty(2)

            # Calculate the first element of the objective function - signed difference in distances to the left and right soft track limits
            objFunc[0] = lLimitLeftSoft - lLimitRightSoft

            # Calculate the second element of the objective function - signed difference in the angles that the soft track limits are widening/
            # narrowing on the left and right sides
            objFunc[1] = (np.interp(sLimitLeftSoft, reducedLimitLeftSoft.sCoords, reducedLimitLeftSoft.AHeadingsFilt)
                          + np.interp(sLimitRightSoft, reducedLimitRightSoft.sCoords, reducedLimitRightSoft.AHeadingsFilt)
                          - (2 * candidateGate.AHeading))

            return objFunc

        def __gateObjFuncMinimize(params: NDArrayFloat1D,
                                  prevGate: Gate,
                                  reducedLimitLeftSoft: CoordinateArray,
                                  reducedLimitRightSoft: CoordinateArray) -> float:
            """
            Internal method to get the objective function value of the candidate
            gate, for the fallback gate placement optimisation method.

            Args:
                params: Optimisation parameters in the form [psi, theta]. psi is
                    the clockwise angle in radians from the previous gate
                    heading to place the new gate. theta is the clockwise
                    heading angle offset in radians of the new gate, relative to
                    psi.
                prevGate: Previous gate, can be a normal track gate, skipped
                    gate, or event gate.
                reducedLimitLeftSoft: Reduced left soft track limit
                    CoordinateArray object.
                reducedLimitRightSoft: Reduced right soft track limit
                    CoordinateArray object.

            Returns:
                Objective function value which is the sum of the distances to
                the left and right soft track limits, plus their absolute
                difference.
            """
            # Get the candidate gate
            candidateGate = __getGateFromParams(params, prevGate)

            # Calculate the candidate gate's intersections with the soft track limits
            lLimitLeftSoft, _ = candidateGate.calcIntersection(reducedLimitLeftSoft, 'limitLeftSoft')
            lLimitRightSoft, _ = candidateGate.calcIntersection(reducedLimitRightSoft, 'limitRightSoft')

            # Calculate the objective function value
            objFunc = lLimitLeftSoft + lLimitRightSoft + abs(lLimitLeftSoft - lLimitRightSoft)

            return objFunc


        def __saveTrackPlot(coordArraysDict: dict[str, CoordinateArray],
                            gates: list[Gate] | np.ndarray[tuple[int], np.dtype[np.object_]],
                            indsBadGates: list[int] | None = None) -> None:
            """
            TODO: Function docstring
            """
            ...

        ## Parse track files to dictionaries ##
        # Attribute BClosedTrack is also set in the function __parseTrackFiles()
        coordArraysDict, eventGatesDict = __parseTrackFiles()

        ## Setup before gate creation loop ##
        # List that will contain Gate objects representing the track gates in order of the direction of travel along the track
        gates: list[Gate] = []
        # Dictionary of lists of the distances along the coordinate arrays of the gate intersections with all coordinate arrays provided, where each
        # index in the list corresponds to the index of the intersecting gate
        sIntersectionsDict: dict[str, list[float]] = {k: [] for k in coordArraysDict.keys()}

        # Create the first gate defined by the first coordinates of the soft track limits, calculating its distances to the soft track limits, and
        # initialising it with the GateCreation event and distances to the soft track limits defined
        lLimitSoft = float(np.linalg.norm(coordArraysDict['limitLeftSoft'].xyzCoords[0] - coordArraysDict['limitRightSoft'].xyzCoords[0]) / 2)
        firstGate = __getGateFromCoords(coordArraysDict['limitLeftSoft'].xyzCoords[0],
                                        coordArraysDict['limitRightSoft'].xyzCoords[0],
                                        Event('GateCreation', 'GateCreation', True, {}),
                                        lLimitSoft,
                                        lLimitSoft)

        # While loop as the calculations are reused if the first gate defined above intersects with the start gate
        sCandidateDict = {}
        BStartGate = False
        BValidFirstGate = False
        while not BValidFirstGate:
            # Align the first gate's heading angle to be within pi of the average initial filtered heading angle of all coordinate arrays
            AFiltAvg = sum([coordArraysDict[key].AHeadingsFilt[0] for key in coordArraysDict.keys()]) / len(coordArraysDict.keys())
            firstGate.AHeading = utils.wrap(firstGate.AHeading, AFiltAvg - np.pi, AFiltAvg + np.pi)

            # Calculate the gate intersections with all the coordinate arrays
            for key, coordArray in coordArraysDict.items():
                if key in ('limitLeftSoft', 'limitRightSoft'):
                    # Soft track limits intersect at 0 distance along its coordinates, from the definition of the first gate
                    sCandidateDict[key] = 0

                else:
                    # If the coordinate array is not a soft track limit, calculate its reduced CoordinateArray object
                    reducedCoordArray = coordArray.getReducedCoordArray(0,
                                                                        -coordArray.sCoords[-1],
                                                                        coordArray.sCoords[-1],
                                                                        coordArray.AHeadingsFilt[0] - REDUCED_HEADING_WINDOW,
                                                                        coordArray.AHeadingsFilt[0] + REDUCED_HEADING_WINDOW)

                    # Calculate the gate intersection with the reduced CoordinateArray object, storing the distance along the coordinate array found
                    _, sCandidateDict[key] = firstGate.calcIntersection(reducedCoordArray, key)

            # Update the first gate's width to follow the GATE_WIDTH_EXTEND constant
            firstGate.updateWidths()

            # If this is the first time in this loop (first gate is not the start gate)
            if not BStartGate:
                if firstGate.xyLine.intersects(eventGatesDict['StartFinish'][0].xyLine):
                    # First gate intersects with the start gate, use the start gate as the first gate, remove it from eventGatesDict and set the flags
                    # to rerun the loop for the start gate
                    firstGate = eventGatesDict['StartFinish'].pop(0)
                    BStartGate = True
                else:
                    # First gate does not intersect with the start gate so it's valid, set the flags to exit the loop
                    BValidFirstGate = True
            else:
                # This loop iteration used the start gate as the first gate, set the flags to exit the loop
                BValidFirstGate = True

        # Append the first gate to the gates list and append the distances along the coordinate arrays of its intersections with the coordinate arrays
        # to the relevant lists in sIntersectionsDict
        gates.append(firstGate)
        for s, key in sCandidateDict.items():
            sIntersectionsDict[key].append(s)

        # Create the final gate used for stopping gate creation
        if self.BClosedTrack:
            # Closed track, use the first gate as the final gate, where the gate only spans the width of the soft track limits and the event attribute
            # set for the final gate of the 'GateCreation' type
            finalGate = Gate(firstGate.xyMidpoint,
                             firstGate.AHeading,
                             Event('GateCreation', 'GateCreation', False, {}),
                             firstGate.lLimitLeftSoft,
                             firstGate.lLimitRightSoft)
        else:
            # Track is not closed, create the final gate from the final coordinates of the soft track limits, with the event attribute set for the
            # final gate of the 'GateCreation' type and with the distances to the soft track limits calculated and defined
            lLimitSoft = float(np.linalg.norm(coordArraysDict['limitLeftSoft'].xyzCoords[-1] - coordArraysDict['limitRightSoft'].xyzCoords[-1]) / 2)
            finalGate = __getGateFromCoords(coordArraysDict['limitLeftSoft'].xyzCoords[-1],
                                            coordArraysDict['limitRightSoft'].xyzCoords[-1],
                                            Event('GateCreation', 'GateCreation', False, {}),
                                            lLimitSoft,
                                            lLimitSoft)

        # Add the 'GateCreation' key and value to eventGatesDict
        eventGatesDict['GateCreation'] = [finalGate]

        ## Gate creation loop ##
        prevGate = firstGate
        BStopGateCreation = False
        while not BStopGateCreation:
            # Get the reduced CoordinateArray objects and their Shapely LineStrings in the 2D [x, y] plane for each of the coordinate arrays, for the
            # expected region around the gate
            reducedCoordArraysDict: dict[str, CoordinateArray] = {}
            for key, coordArray in coordArraysDict.items():
                sPrev = sIntersectionsDict[key][-1]
                APrev = gates[-1].AHeading
                reducedCoordArraysDict[key] = coordArray.getReducedCoordArray(sPrev,
                                                                              sPrev,
                                                                              sPrev + REDUCED_DISTANCE_WINDOW,
                                                                              APrev - REDUCED_HEADING_WINDOW,
                                                                              APrev + REDUCED_HEADING_WINDOW)

            # Optimise the new gate placement using root-finding method
            x0 = np.array([0, 0])
            args = (prevGate, reducedCoordArraysDict['limitLeftSoft'], reducedCoordArraysDict['limitRightSoft'])
            sol = scipy.optimize.root(__gateObjFuncRoot, x0, args, method='hybr')

            # Get the optimised candidate gate and calculate its intersections with the soft track limits
            candidateGate = __getGateFromParams(sol.x, prevGate)
            lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedCoordArraysDict['limitLeftSoft'], 'limitLeftSoft')
            lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedCoordArraysDict['limitRightSoft'], 'limitRightSoft')

            # Check if the root-finding method failed
            if not sol.success or lLimitLeftSoft >= GATE_MAX_WIDTH / 2 or lLimitRightSoft >= GATE_MAX_WIDTH / 2:
                # Root-finding failed, fall back to optimisation using minimise, then get the optimised candidate gate and calculate its intersections
                # with the soft track limits
                sol = scipy.optimize.minimize(__gateObjFuncMinimize, x0, args, method='BFGS')
                candidateGate = __getGateFromParams(sol.x, prevGate)
                lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedCoordArraysDict['limitLeftSoft'], 'limitLeftSoft')
                lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedCoordArraysDict['limitRightSoft'], 'limitRightSoft')

            # Check if the minimise method failed
            if not sol.success or lLimitLeftSoft >= GATE_MAX_WIDTH / 2 or lLimitRightSoft >= GATE_MAX_WIDTH / 2:
                # Minimise method failed, save the track plot for debugging and raise a ValueError
                gates.append(candidateGate)
                indsBadGates = [-1]
                __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                raise ValueError(f"Both methods for optimisation of new gate placement failed, see track plot in {trackPath}")

            # Calculate the rest of the gate attributes
            sCandidateDict = {}
            for key, reducedCoordArray in reducedCoordArraysDict.items():
                if key == 'limitLeftSoft':
                    # Already calculated the intersection with the left soft track limits above
                    sCandidateDict[key] = sLimitLeftSoft
                elif key == 'limitRightSoft':
                    # Already calculated the intersection with the right soft track limits above
                    sCandidateDict[key] = sLimitRightSoft
                else:
                    # Calculate the intersection with the coordinate array and store the distance along the coordinate array of the intersection
                    sCandidateDict[key] = candidateGate.calcIntersection(reducedCoordArraysDict[key], key)

            # Update the gate width to follow the GATE_EXTEND_WIDTH constant and recalculate the gate midpoint
            candidateGate.updateWidths()
            candidateGate.recalcMidpoint()

            # Find the event gates contained in the track segment from the previous gate to this candidate gate
            xyLinePrevToCandidateMidpoint = shapely.LineString([prevGate.xyMidpoint, candidateGate.xyMidpoint])
            AHeadingAvg = (prevGate.AHeading + candidateGate.AHeading) / 2
            eventGatesFlat = []
            for eventGatesList in eventGatesDict.values():
                for eventGate in eventGatesList:
                    eventGatesFlat.append(eventGate)
            eventGatesContained: list[Gate] = []
            sEventGatesContained: list[dict[str, float]] = []
            for eventGate in eventGatesFlat:
                # Calculate the heading angle difference compared to the average heading angle of the previous and candidate gates
                AHeadingDiff = utils.wrap(eventGate.AHeading - AHeadingAvg, -np.pi, np.pi)

                # Check if the line from the previous gate midpoint to the candidate gate midpoint intersects with the event gate, and if the event
                # gate has a similar heading angle to the average heading angle of the previous and candidate gate
                if xyLinePrevToCandidateMidpoint.intersects(eventGate.xyLine) and abs(AHeadingDiff) <= HEADING_ANGLE_THRESHOLD:
                    # Event gate is contained within the track segment from the previous gate to the candidate gate, append it to the
                    # eventGatesContained list
                    eventGatesContained.append(eventGate)

                    # Calculate the intersections with the coordinate arrays and store the distances along the coordinate arrays of the intersections
                    sEventGatesContained.append({})
                    for key, reducedCoordArray in reducedCoordArraysDict.items():
                        _, sEventGatesContained[-1][key] = eventGate.calcIntersection(reducedCoordArraysDict[key], key)

                    # Align the event gate's heading angle to be within pi of the average heading angle of the previous and candidate gates
                    eventGate.AHeading = utils.wrap(eventGate.AHeading, AHeadingAvg - np.pi, AHeadingAvg + np.pi)

                    # Update the event gate width to follow the GATE_EXTEND_WIDTH constant and recalculate the event gate midpoint
                    eventGate.updateWidths()
                    eventGate.recalcMidpoint()

            # Sort the eventGatesContained and sEventGatesContained lists in order of ascending distance of the projection of their midpoint onto the
            # line from the previous gate midpoint to the candidate gate midpoint
            zipList = zip(eventGatesContained, sEventGatesContained)
            zipListSorted = sorted(zipList, key=lambda x: xyLinePrevToCandidateMidpoint.project(shapely.Point(x[0].xyMidpoint)))
            eventGatesContained, sEventGatesContained = zip(*zipListSorted)

            # Check if both the start and finish event gates are in the eventGatesContained list, and swap them if so
            indStart = None
            indFinish = None
            for i, eventGate in enumerate(eventGatesContained):
                # Find the indexes of the start and finish gates
                if eventGate.event.type == 'StartFinish' and eventGate.event.BStart:
                    indStart = i
                elif eventGate.event.type == 'StartFinish' and not eventGate.event.BStart:
                    indFinish = i

                if indStart is not None and indFinish is not None:
                    # Both start and finish gates are in the eventGatesContained list, swap them in both eventGatesContained and sEventGatesContained
                    eventGatesContained[indStart], eventGatesContained[indFinish] = eventGatesContained[indFinish], eventGatesContained[indStart]
                    sEventGatesContained[indStart], sEventGatesContained[indFinish] = sEventGatesContained[indFinish], sEventGatesContained[indStart]
                    break

            # Create a flag for whether the candidate gate is valid, initialising to true
            BValidGate = True

            if len(eventGatesContained) > 0:
                # Check if any of the event gates contained in the track segment have the event to finish the gate creation loop
                BStopEventGates = [eventGate.event.type == 'GateCreation' and eventGate.event.BStart for eventGate in eventGatesContained]
                if any(BStopEventGates):
                    # Finish gate creation event gate is contained by the track segment, set the flags to mark the candidate gate as invalid and stop
                    # gate creation
                    BValidGate = False
                    BStopGateCreation = True

                    # Check if the finish gate creation event gate is the last gate in the eventGatesContained list
                    indStopEventGate = BStopEventGates.index(True)
                    if indStopEventGate == len(BStopEventGates):
                        # Finish gate creation event gate is the last gate in the eventGatesContained list
                        if len(eventGatesContained) > 1:
                            # Check if the finish gate creation event gate intersects with the 2nd last gate in the eventGatesContained list
                            if eventGatesContained[-1].xyLine.intersects(eventGatesContained[-2].xyLine):
                                # Intersection found, remove the finish gate creation event gate from eventGatesContained and sEventGatesContained
                                del eventGatesContained[-1]
                                del sEventGatesContained[-1]

                    else:
                        # Still more event gates after the finish gate creation event gate, remove them from eventGatesContained and
                        # sEventGatesContained (will result in an exception being raised later due to gate creation stopping before all event gates
                        # were added)
                        del eventGatesContained[indStopEventGate + 1:]
                        del sEventGatesContained[indStopEventGate + 1:]

                # Check if consecutive event gates intersect with each other at a single point (here it is acceptable to intersect over a line, as
                # event gates can share the same line)
                for i in range(len(eventGatesContained) - 1):
                    intersection = shapely.intersection(eventGatesContained[i].xyLine, eventGatesContained[i + 1].xyLine)
                    if isinstance(intersection, shapely.Point) and not intersection.is_empty:
                        # Point intersection found between consecutive event gates, save the track plot for debugging and raise a ValueError
                        indsBadGates = [len(gates), len(gates) + i]
                        for eventGate in eventGatesContained:
                            gates.append(eventGate)
                        __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                        raise ValueError(f"Consecutive event gates intersect, see track plot in {trackPath}: possible fixes are spacing out the "
                                         f"event gates more, reducing GATE_EXTEND_WIDTH, increasing HEADING_ANGLE_THRESHOLD")

                # Check if the first event gate contained by the track segment intersects with the last gate in the gates list, and if so, remove the
                # last gate in the gates list and repeat until the first event gate no longer intersects with the last gate in the gates list
                BIntersects = True
                while BIntersects and len(gates) > 0:
                    # Check if the first event gate contained by the track segment intersects (at all) with the last gate in the gates list
                    if eventGatesContained[0].xyLine.intersects(gates[-1].xyLine):
                        # Intersection found
                        BIntersects = True
                        if gates[-1].event is None:
                            # Last gate in the gates list is not an event gate, remove it from the gates list and remove the last elements from all
                            # the lists in sIntersectionsDict
                            del gates[-1]
                            for sIntersections in sIntersectionsDict.values():
                                del sIntersections[-1]
                        else:
                            # Last gate in the gates list is an event gate
                            gates.append(eventGatesContained[0])
                            indsBadGates = [-2, -1]
                            __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                            raise ValueError(f"Consecutive event gates intersect, see track plot in {trackPath}: possible fixes are spacing out the "
                                             f"event gates more, reducing GATE_EXTEND_WIDTH, increasing HEADING_ANGLE_THRESHOLD")
                    else:
                        # No intersection found
                        BIntersects = False

                # Check if the last event gate contained by the track segment intersects with the candidate gate
                if eventGatesContained[-1].xyLine.intersects(candidateGate.xyLine):
                    # Intersection found, mark the candidate gate as invalid
                    BValidGate = False

                # Append the event gates and the distances along the coordinate arrays of their intersections with the coordinate arrays to the
                # relevant lists, and remove them from eventGatesDict
                for i, eventGate in enumerate(eventGatesContained):
                    gates.append(eventGate)
                    for key in sEventGatesContained[i].keys():
                        sIntersectionsDict[key].append(sEventGatesContained[i][key])
                    eventGatesDict[eventGate.event.type].remove(eventGate)

            # Check if the candidate gate intersects with the last gate in the gates list
            if len(gates) > 0:
                if candidateGate.xyLine.intersects(gates[-1].xyLine):
                    # Intersection found, mark the candidate gate as invalid
                    BValidGate = False

            # If the candidate gate is still valid after all the checks, append it and the distances along the coordinate arrays of its intersections
            # with the coordinate arrays to the relevant lists
            if BValidGate:
                gates.append(candidateGate)
                for key in sCandidateDict.keys():
                    sIntersectionsDict[key].append(sCandidateDict[key])

            # Update prevGate to be the candidate gate
            prevGate = candidateGate

        # Postprocessing and validation
        # Remove the first or last gate in the gates list if they have the event attribute with the GateCreation type
        for i in (0, -1):
            if gates[i].event is not None:
                if gates[i].event.type == 'GateCreation':
                    del gates[i]
                    for sIntersections in sIntersectionsDict.values():
                        del sIntersections[i]

        # Set the gates attribute
        self.gates = np.array(gates)

        # Find the indexes of the start and finish gates
        indStart = -1
        indFinish = -1
        for i, gate in enumerate(gates):
            if gate.event is not None:
                if gate.event.type == 'StartFinish':
                    if gate.event.BStart:
                        indStart = i
                    else:
                        indFinish = i
                    break

        if self.BClosedTrack:
            # Track is closed, roll the order of the gates so that the finish gate is the last index in the lists
            self.gates = np.roll(self.gates, -indFinish - 1)
        elif indStart < indFinish:
            # Track is not closed, and the start gate comes before the finish gate
            indsBadGates = [indStart, indFinish]
            __saveTrackPlot(coordArraysDict, self.gates, indsBadGates)
            raise ValueError(f"Track is not closed but the finish gate is before the start gate, see track plot in {trackPath}")

        # Check if there are any remaining event gates that haven't been included in the track
        if any(eventGatesDict.values()):
            # There are event gates that haven't been included in the track
            indsBadGates = []
            for eventGatesList in eventGatesDict.values():
                for eventGate in eventGatesList:
                    self.gates = np.append(self.gates, eventGate)
                    indsBadGates.append(len(self.gates))
            __saveTrackPlot(coordArraysDict, self.gates, indsBadGates)
            raise ValueError(f"Gate creation stopped before all event gates were added, see track plot in {trackPath}: event gates not added are "
                             f"{[f"{gate.event.name} (Start: {gate.event.BStart})" for gate in self.gates[indsBadGates]]}")

        ## Track mesh creation ##
        # Calculate [x, y, z] coordinates at the left and right coordinates of the gates to "artificially" expand the track mesh area and avoid NaNs
        nGates = len(self.gates)
        xyzGatesLeft = np.empty((nGates, 3))
        xyzGatesRight = np.empty((nGates, 3))
        for i, gate in enumerate(self.gates):
            # Get the [x, y] coordinate at the left and right coordinates of the gate
            xyzGatesLeft[i][:2] = gate.xyLine.xy[:, 0]
            xyzGatesRight[i][:2] = gate.xyLine.xy[:, 1]

            # Linearly extrapolate the z coordinate at the left and right coordinates of the gate from the z coordinates at the hard track limits
            xp = [-gate.lLimitLeftHard, gate.lLimitRightHard]
            fp = [float(np.interp(sIntersectionsDict['limitLeftHard'],
                                  coordArraysDict['limitLeftHard'].sCoords,
                                  coordArraysDict['limitLeftHard'].xyzCoords[:, 2])),
                  float(np.interp(sIntersectionsDict['limitRightHard'],
                                  coordArraysDict['limitRightHard'].sCoords,
                                  coordArraysDict['limitRightHard'].xyzCoords[:, 2]))]
            xyzGatesLeft[i][2] = utils.linearInterpExtrap(-gate.lLeft, xp, fp)
            xyzGatesRight[i][2] = utils.linearInterpExtrap(gate.lRight, xp, fp)

        # Find unused keys in coordArraysDict
        keyGateLeft = 0
        while str(keyGateLeft) not in coordArraysDict.keys():
            keyGateLeft += 1
        keyGateLeft = str(keyGateLeft)
        keyGateRight = 1
        while str(keyGateRight) not in coordArraysDict.keys():
            keyGateRight += 1
        keyGateRight = str(keyGateRight)

        # Create CoordinateArray objects from the extrapolated gate left and right coordinates, and add them to coordinateArraysDict - note that the
        # heading angles of these coordinate arrays does not matter
        coordArraysDict[keyGateLeft] = CoordinateArray(xyzGatesLeft, self.BClosedTrack)
        coordArraysDict[keyGateRight] = CoordinateArray(xyzGatesRight, self.BClosedTrack)

        # Create the track mesh, which is an array Scipy multivariate interpolators local to the area around their index's gate
        self.mesh = np.empty(nGates)
        self.__indsPrevNext = np.empty((nGates, 2))  # Internal attribute storing [iPrev, iNext] for each gate (indexes of the previous and next gates
                                                     # with different midpoints to the current gate), to avoid re-calculation in calcTrackZ()
        for i, gate in enumerate(self.gates):
            # Create an array that will contain all the [x, y, z] coordinates local to the area around the gate
            xyzCoords = np.array([coordArraysDict[keyGateLeft].xyzCoords[i],
                                  coordArraysDict[keyGateRight].xyzCoords[i]])

            # Find the indexes corresponding to the previous and next gate with a different midpoint to the current gate, and add all the left and
            # right [x, y, y] coordinates of the gates contained between them (inclusive)
            iPrev = i
            BFoundPrev = False
            while not BFoundPrev:
                if not self.BClosedTrack and iPrev == 0:
                    # Reached the start of the track
                    break
                else:
                    iPrev -= 1
                    xyzCoords = np.vstack((xyzCoords, coordArraysDict[keyGateLeft].xyzCoords[iPrev], coordArraysDict[keyGateRight].xyzCoords[iPrev]))
                if self.gates[iPrev].xyMidpoint != gate.xyMidpoint:
                    BFoundPrev = True
            iNext = i
            BFoundNext = False
            while not BFoundNext:
                if not self.BClosedTrack and iNext == nGates - 1:
                    # Reached the end of the track
                    break
                else:
                    iNext += 1
                    xyzCoords = np.vstack((xyzCoords, coordArraysDict[keyGateLeft].xyzCoords[iNext], coordArraysDict[keyGateRight].xyzCoords[iNext]))
                if self.gates[iNext].xyMidpoint != gate.xyMidpoint:
                    BFoundNext = True
            self.__indsPrevNext[i] = np.array([iPrev, iNext])

            # Get the [x, y, z] coordinates of all the reduced coordinate arrays (reduced to be between the previous and next gates)
            for key, coordArray in coordArraysDict.items():
                xyzCoords = np.vstack((xyzCoords, coordArray.getReducedCoordArray(sIntersectionsDict[key][i],
                                                                              sIntersectionsDict[key][iPrev],
                                                                              sIntersectionsDict[key][iNext],
                                                                              -1e9,
                                                                              1e9).xyzCoords))  # A track shouldn't have 159 million spirals

            # Create the interpolator for the region around this gate
            self.mesh[i] = scipy.interpolate.LinearNDInterpolator(xyzCoords[:, 0:2], xyzCoords[:, 2])

        ## Saving ##
        with open(os.path.join(trackPath, TRACK_PKL_FILENAME), 'wb') as pklFile:
            pkl.dump(self, pklFile, protocol=pkl.HIGHEST_PROTOCOL)
        __saveTrackPlot(coordArraysDict, self.gates)

    def getTrackZ(self,
                   xy: list[float] | NDArrayFloat1D,
                   indGate: int,
                   BReturnNaN: bool = False) -> float:
        """
        Calculates the z coordinate of the track at the specified [x, y]
        coordinate using the track mesh (local z coordinate interpolators).

        Chooses the closest interpolator such that the [x, y] coordinate is
        within the half-step backwards/forwards to the previous/next gate. If
        the [x, y] coordinate is outside the interpolation region, will return 0
        or NaN, depending on the argument BReturnNaN.

        Args:
            xy: Coordinate to calculate the z coordinate of the track at, in the
                2D plane [x, y].
            indGate: Index of the closest/most recent gate. This determines
                which local track region the xy coordinate is in.
            BReturnNaN: Whether to return NaN if the xy coordinate is outside
                the interpolation region, or to sanitise it and return 0.

        Returns:
            z coordinate at the specified [x, y] coordinate.
        """
        # Store previous and next gate indexes for easier access
        indGatePrev = self.__indsPrevNext[indGate][0]
        indGateNext = self.__indsPrevNext[indGate][1]

        # Calculate the midpoints and left-right direction vectors of the virtual gates at the half-step between the gate and the previous/next gate
        xyLineHalfStepPrev = (((self.gates[indGate].xyMidpoint + self.gates[indGatePrev].xyMidpoint) / 2)
                              + utils.rotateVectorHeading(np.array([0, 1]),
                                                          (self.gates[indGate].AHeading + self.gates[indGatePrev].AHeading + np.pi) / 2))
        xyLineHalfStepNext = (((self.gates[indGate].xyMidpoint + self.gates[indGateNext].xyMidpoint) / 2)
                              + utils.rotateVectorHeading(np.array([0, 1]),
                                                          (self.gates[indGate].AHeading + self.gates[indGateNext].AHeading + np.pi) / 2))

        # Check if the gate index specified is correct for the specified coordinate
        if utils.getSideOfLine(xy, xyLineHalfStepPrev[0], xyLineHalfStepPrev[1]) > 0:
            # Specified coordinate is behind the virtual gate at the half-step between the gate and the previous gate
            z = self.getTrackZ(xy, indGatePrev, BReturnNaN)
        elif utils.getSideOfLine(xy, xyLineHalfStepNext[0], xyLineHalfStepNext[1]) < 0:
            # Specified coordinate is ahead of the virtual gate at the half-step between the gate and the next gate
            z = self.getTrackZ(xy, indGateNext, BReturnNaN)
        else:
            # Index is correct for the specified coordinate
            z = self.mesh[indGate](xy[0], xy[1])

        # Sanitise the z coordinate
        if np.isnan(z) and not BReturnNaN:
            z = 0

        return z

    def getTrackNormal(self,
                        xy_xyz: list[float] | NDArrayFloat1D,
                        indGate: int) -> NDArrayFloat1D:
        """
        Calculates the normal vector to the track at the specified [x, y] or
        [x, y, z] coordinate from forward differencing of the z coordinate.

        If xy_xyz is in the form [x, y, z], then the z coordinate will be used
        as the unperturbed track z coordinate (i.e. skipping the calculation).

        Note that forward differencing is used as the SciPy multivariate
        interpolators don't seem to return their gradients. If the forward
        differencing in any direction gets a NaN z coordinate, will use
        backwards differencing for that direction.

        Args:
            xy_xyz: Coordinate to calculate the track normal vector at, in the
                form [x, y] or [x, y, z].
            indGate: Index of the closest/most recent gate. This determines
                which local track region the xy coordinate is in.

        Returns:
            Upwards-facing normal vector to the track at the specified
            coordinate, scaled to a magnitude of 1. If the specified coordinate
            is outside the interpolation region of the track mesh, returns the
            normal pointing directly upwards [0, 0, 1].
        """
        direction = 1

        # Get unperturbed [x, y] and z coordinates
        if len(xy_xyz) < 3:
            xy = np.array(xy_xyz)
            z = self.getTrackZ(xy_xyz, indGate, True)
        else:
            xy = np.array(xy_xyz[0:2])
            z = xy_xyz[2]

        # Check that unperturbed z coordinate is within the interpolation region of the track mesh
        if not np.isnan(z):
            xyz = np.append(xy_xyz, z)
        else:
            # Unperturbed z coordinate is outside the interpolation region of the track mesh, return failed track normal
            return np.array([0, 0, 1])

        # Calculate coordinate perturbed in the x direction
        xyPerturbX = xy + np.array([PERTURB_DISTANCE, 0])
        zPerturbX = self.getTrackZ(xyPerturbX, indGate, True)
        if np.isnan(zPerturbX):
            # Forward-perturbed coordinate in the x direction is outside the interpolation region of the track mesh, try backwards-perturb
            direction *= -1
            xyPerturbX = xy - np.array([PERTURB_DISTANCE, 0])
            zPerturbX = self.getTrackZ(xyPerturbX, indGate, True)
            if np.isnan(zPerturbX):
                # Backwards-perturbed coordinate in the x direction is also outside the interpolation region, return failed track normal
                return np.array([0, 0, 1])
        xyzPerturbX = np.append(xyPerturbX, zPerturbX)

        # Calculate coordinate perturbed in the y direction
        xyPerturbY = xy + np.array([0, PERTURB_DISTANCE])
        zPerturbY = self.getTrackZ(xyPerturbY, indGate, True)
        if np.isnan(zPerturbX):
            # Forward-perturbed coordinate in the y direction is outside the interpolation region of the track mesh, try backwards-perturb
            direction *= -1
            xyPerturbY = xy - np.array([0, PERTURB_DISTANCE])
            zPerturbY = self.getTrackZ(xyPerturbY, indGate, True)
            if np.isnan(zPerturbY):
                # Backwards-perturbed coordinate in the y direction is also outside the interpolation region, return failed track normal
                return np.array([0, 0, 1])
        xyzPerturbY = np.append(xyPerturbY, zPerturbY)

        # Calculate the vectors to the perturbed coordinates
        xyzVecX = xyzPerturbX - xyz
        xyzVecY = xyzPerturbY - xyz

        # Calculate the track normal as the (scaled) cross product of the vectors to the 2 perturbed coordinates
        xyzTrackNormal = np.cross(xyzVecY, xyzVecX)
        xyzTrackNormal /= np.linalg.norm(xyzTrackNormal) * direction

        return xyzTrackNormal

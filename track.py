"""
The track module is responsible for defining the track on which a trajectory
can be created and optimised.

This includes the Track class, as well as the CoordinateArray, Event and Gate
classes used to generate and define the track.
"""
# Python standard libraries
import os
import json
import pickle as pkl
from typing import Literal
from dataclasses import dataclass

# External libraries
import scipy
import shapely
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from Utils.typeAliases import Any, NDArrayFloat1D, NDArrayFloat2D, NDArrayInt1D
import Utils.utils as utils

# Filename constants
TRACK_PKL_FILENAME = "Track.pkl"
TRACK_PLOT_FILENAME = "TrackPlot.png"
LIMIT_LEFT_SOFT_FILENAME = "xyzLimitLeftSoft.csv"
LIMIT_RIGHT_SOFT_FILENAME = "xyzLimitRightSoft.csv"
LIMIT_LEFT_HARD_FILENAME = "xyzLimitLeftHard.csv"
LIMIT_RIGHT_HARD_FILENAME = "xyzLimitRightHard.csv"

# CoordinateArray constants
LP_FILT_SPATIAL_FREQ = 0.01                     # Low-pass cutoff spatial frequency (cycles/m) - 0.01 to 0.05 seem good (100 to 20 m wavelengths)
LP_FILT_ORDER = 1                               # Order of the low-pass filter

# Event constants
CUSTOM_EVENT_TYPES = Literal['TODO']            # List of valid event types for custom events TODO: Implement custom events and add them here
INTERNAL_EVENT_TYPES = Literal['GateCreation',  # Valid event types for internal events
                               'StartFinish']

# Gate constants
GATE_MAX_WIDTH = 1000                           # Maximum width of the gate, used as the initial gate width during gate creation

GATE_EXTEND_WIDTH_SOFT = 25                     # Additional width beyond the corresponding soft/hard track limits to extend the gate width on each
GATE_EXTEND_WIDTH_HARD = 10                     # side, to allow suitable penalties to be calculated if track limits are violated (m)
                                                # The gate width will be extended to whichever is the more limiting criteria - GATE_EXTEND_WIDTH_SOFT
                                                # can be used to reduce gate skipping in tight corners if the hard track limits are significantly
                                                # further away
                                                #  - Higher gives more robustness, allowing a trajectory that exceeds track limits to still be solved
                                                #  - Too high can cause gates to be skipped in tight corners due to overlapping, losing track limits
                                                #    resolution

GATE_SIMILARITY_THRESHOLD = 1e-6                # Threshold for difference in gate midpoints or gate heading angles, where if both are met, the gate
                                                # is considered to lie on the same line (for floating point precision)

# Track generation constants
CLOSED_TRACK_THRESHOLD_DISTANCE = 20            # Maximum distance (as the crow flies) from the start to finish coordinates of the provided soft track
                                                # limits to consider the track closed, when using the automatic logic (m)

HEADING_ANGLE_THRESHOLD = np.pi / 4             # Threshold for the difference in heading angles (where the difference is wrapped from -pi to pi then
                                                # the absolute value is taken), above which to consider the heading angles as "similar"

GATE_STEP_DISTANCE = 5                          # Distance between each consecutive gate for gate creation, unless overridden by an event gate or the
                                                # gate is skipped as it overlaps with neighbouring gates (m)
                                                #  - Higher gives more resolution for determining track limits
                                                #  - Too high may excessively slow track and trajectory generation

REDUCED_DISTANCE_WINDOW = 500                   # Window on each side for the distance bounds of the reduced coordinate array for gate creation (m)

REDUCED_HEADING_WINDOW = 3 / 4 * np.pi          # Window on each side for the heading angle bounds of the reduced coordinate array for gate creation
                                                # (radians)

# Track plot constants
TRACK_PLOT_MARGIN = 0.75                        # Margin around the track plot axes for the axis scales (inches)
TRACK_PLOT_SCALE = 15                           # Scale of the track plot, for adjusting  size of the plot labels, line thicknesses etc. (m/inch)
TRACK_PLOT_SPATIAL_RESOLUTION = 10              # Spatial resolution of the track plot (pixels/m)
TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION = 1          # Spatial resolution of the z coordinate map for the track plot (z coordinate points/m)
TRACK_PLOT_ZMAP_CONTOUR_INTERVALS = 2           # Spacing between contour levels (m)
TRACK_PLOT_OVERWRITE_Z = True                   # If true, overwrites the z coordinates with the values from further along the track, if false then
                                                # keeps z coordinates that were from earlier points on the track

# Track normal vector constants
PERTURB_DISTANCE = 0.1                          # Radius of the circle around the specified coordinate used to sample perturbed coordinates to
                                                # calculate the track normal vector


class CoordinateArray:
    """
    Array of coordinates and its derived attributes.

    Used for track generation to store the coordinates defining the track limits
    and track mesh.

    Has methods:
        - rotateHeadings(theta)
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
        # Make sure this is a closed coordinate array if the track is closed
        # (last coordinate equal to the first coordinate, at least in the 2D plane [x, y])
        if BClosedTrack and not all(xyzCoords[-1][:2] == xyzCoords[0][:2]):
            xyzCoords = np.vstack((xyzCoords, xyzCoords[0]))

        # Get the indexes that omit consecutive duplicates from the coordinate array, and remove the consecutive duplicates from the coordinate array,
        # then set the xyzCoords attribute
        inds = utils.getIndsWithoutConsecutiveDuplicates(xyzCoords, axis=0)
        self.xyzCoords = xyzCoords[inds]

        # Generate the Shapely LineString of the coordinates in the 2D plane [x, y]
        self.xyLine = shapely.LineString(self.xyzCoords[:, :2])

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
            dxyzCoords = np.diff(xyzCoords[:, :2], axis=0)
            dsCoords = np.linalg.norm(dxyzCoords, axis=1)
            self.sCoords = np.append(0, np.cumsum(dsCoords))

            # Calculate AHeadings - unwrapped heading angle along the coordinates, calculated as the average of the forward and backward angles
            AHeadingsTemp = np.unwrap(utils.getHeading(dxyzCoords))
            AHeadingsForwards = np.append(AHeadingsTemp[0], AHeadingsTemp)
            AHeadingsBackwards = np.append(AHeadingsTemp, AHeadingsTemp[-1])
            self.AHeadings = (AHeadingsForwards + AHeadingsBackwards) / 2

            # Calculate AHeadingsFilt - low-pass filtered AHeadings
            # Resample to regular intervals (10x the low-pass filter cutoff frequency), then filter, then resample back to original signal base
            AHeadingsResampled, sCoordsResampled = utils.resample(self.AHeadings, self.sCoords, LP_FILT_SPATIAL_FREQ * 10, True)
            AHeadingsResampled[-1] = self.AHeadings[-1]
            AHeadingsFiltResampled = utils.filt(AHeadingsResampled, LP_FILT_SPATIAL_FREQ * 10, 'low', LP_FILT_SPATIAL_FREQ, LP_FILT_ORDER)
            self.AHeadingsFilt = np.interp(self.sCoords, sCoordsResampled, AHeadingsFiltResampled)

    def rotateHeadings(self,
                       theta: float) -> None:
        """
        Offsets the attributes AHeadings and AHeadingsFilt by theta.

        Use this to correct the calculated heading angle attributes to ensure
        that all CoordinateArray objects are in the same rotation.

        Args:
            theta: Heading angle in radians, clockwise, to offset the AHeadings
                and AHeadingsFilt attributes.
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
        until a distance or heading bound is reached. The coordinate on the
        bound is then interpolated for both the start and finish points of the
        reduced coordinate array. The coordinates are ordered in the direction
        of travel.

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
                [ALower, AUpper].
            IndexError: No coordinates within the bounds specified.
        """
        # Automatically determine if the track is closed
        BClosedTrack = all(self.xyzCoords[-1] == self.xyzCoords[0])

        # Validate that the coordinate at the reference distance is within the heading angle bounds
        if not BClosedTrack:
            ARef = np.interp(sRef, self.sCoords, self.AHeadingsFilt)
            if not ALower <= ARef <= AUpper:
                raise ValueError(f"Invalid reference distance or heading angle bounds for a non-closed track: "
                                 f"heading angle {ARef} at sRef {sRef} is outside the heading angle bounds [{ALower}, {AUpper}]")

        # Get the indexes that satisfy both the distance and heading bounds
        # Get arrays of all the indexes forward and backward of the reference distance
        nCoords = len(self.sCoords)
        startInd = np.interp(sRef, self.sCoords, np.arange(nCoords))
        indsForward = np.arange(np.ceil(startInd), nCoords, 1, dtype=int)
        indsBackward = np.arange(0, np.floor(startInd) + 1, 1, dtype=int)

        # If the track is closed, include the indexes wrapping around back to the start index (i.e. contain all indexes)
        # In both cases, flip the backwards arrays so that incrementing the index on these arrays equates to moving backwards around the track
        # Also get the wrapped distance bounds (if it is a closed track)
        if BClosedTrack:
            indsForward = np.append(indsForward, indsBackward)
            indsBackward = np.flip(indsForward)
            sLowerWrapped = utils.wrap(sLower, 0, self.sCoords[-1])
            sUpperWrapped = utils.wrap(sUpper, 0, self.sCoords[-1])
        else:
            indsForward = indsForward
            indsBackward = np.flip(indsBackward)
            sLowerWrapped = sLower
            sUpperWrapped = sUpper

        def __inDistanceBounds(s: float) -> bool:
            """
            Internal function to check whether the distance s along the
            coordinate array is within the distance bounds.

            Args:
                s: Distance along the coordinate array to check.

            Returns:
                Boolean of whether the distance s is within the distance bounds.
            """
            if BClosedTrack and (sLower < 0 or sUpper > self.sCoords[-1] or sLower > sUpper):
                # Closed track and the bounds will wrap, or closed track and the bounds are already wrapped
                if sLowerWrapped <= sUpperWrapped:
                    # Wrapped bounds cover the whole track
                    return True
                else:
                    # Wrapped bounds don't cover the whole track - and due to the wrapping, sLowerWrapped > sUpperWrapped
                    return sLowerWrapped <= s or s <= sUpperWrapped
            else:
                # Not a closed track or the bounds don't wrap
                return sLower <= s <= sUpper

        # Find the indexes within the heading angle bounds in the forwards and backwards direction
        # In each direction, stop once the filtered heading angle exceeds either the distance or heading angle bounds
        indsValid = []
        AHeadingsFiltWrapped = utils.wrap(self.AHeadingsFilt, (ALower + AUpper) / 2 - np.pi, (ALower + AUpper) / 2 + np.pi)
        for n, inds in enumerate((indsBackward, indsForward)):
            for i in inds:
                if __inDistanceBounds(self.sCoords[i]) and ALower <= AHeadingsFiltWrapped[i] <= AUpper:
                    # Both the distance and heading angle bounds are satisfied
                    indsValid.append(i)
                else:
                    # Exceeded distance and/or heading bounds
                    break

        # Validate that there are coordinates within the bounds specified, and remove duplicate valid indexes
        if len(indsValid) < 1:
            raise IndexError("No coordinates within the bounds specified")
        indsValid = utils.removeConsecutiveDuplicates(sorted(indsValid))

        # If the track is closed, roll the valid indexes array such that the indexes are continuously incrementing by 1, except for the wrap back to 0
        if BClosedTrack:
            dindsValid = np.diff(indsValid)
            if len(dindsValid) > 0:
                if np.max(dindsValid) > 1:
                    offset = np.where(dindsValid > 1)[0][0] + 1
                    indsValid = np.roll(indsValid, -offset)

        def __getCoordOnBound(BStart: bool) -> tuple[NDArrayFloat1D, float, float, float]:
            """
            Internal function to calculate the coordinate on the bound limiting
            the start or finish of the reduced coordinate array.

            If the track is closed or the bound lies within the coordinate
            array, uses linear interpolation. Otherwise, if the bound lies
            beyond the defined coordinate array, extrapolates using the first/
            last low-pass filtered heading angle.

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

            # Get the next index if the coordinate array continued in this direction
            nextInd = indsValid[indsIndex] - 1 if BStart else indsValid[indsIndex] + 1
            if BClosedTrack:
                nextInd: int = int(utils.wrap(nextInd, 0, len(self.sCoords)))

            if not __inDistanceBounds(self.sCoords[nextInd]):
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
                AClosest = self.AHeadingsFilt[indsValid[indsIndex]]
                ALowerWrapped = utils.wrap(ALower, AClosest - np.pi, AClosest + np.pi)
                AUpperWrapped = utils.wrap(AUpper, AClosest - np.pi, AClosest + np.pi)
                AFiltBound = ALowerWrapped if abs(AClosest - ALowerWrapped) < abs(AClosest - AUpperWrapped) else AUpperWrapped

                # Get the indexes surrounding this heading bound - by definition, the coordinate on the heading bound is interpolated
                if indsIndex == 0:
                    indLower = indsValid[indsIndex] - 1
                    indUpper = indsValid[indsIndex]
                else:
                    indLower = indsValid[indsIndex]
                    indUpper = indsValid[indsIndex] + 1 if indsValid[indsIndex] + 1 < len(self.AHeadingsFilt) - 1 else 0

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
        sCoords = np.hstack((sStart, self.sCoords[indsValid], sFinish))
        AHeadings = np.hstack((AStart, self.AHeadings[indsValid], AFinish))
        AHeadingsFilt = np.hstack((AFiltStart, self.AHeadingsFilt[indsValid], AFiltFinish))

        # Check if the indexes wrapped
        ds = np.diff(sCoords, prepend=sCoords[0])
        indWrap = np.argmin(ds)
        if ds[indWrap] < 0:
            # Indexes wrapped, offset the values at the wrapped indexes so that the coordinate array is continuous
            AOffset = round((self.AHeadingsFilt[-1] - self.AHeadingsFilt[0]) / (2 * np.pi)) * (2 * np.pi)
            if sRef < sCoords[0]:
                # Wrapped going backwards
                sCoords[:indWrap] -= self.sCoords[-1]
                AHeadings[:indWrap] -= AOffset
                AHeadingsFilt[:indWrap] -= AOffset
            else:
                # Wrapped going forwards
                sCoords[indWrap:] += self.sCoords[-1]
                AHeadings[indWrap:] += AOffset
                AHeadingsFilt[indWrap:] += AOffset

        # Return a new CoordinateArray instance initialised with its attributes provided
        # Note that any consecutive duplicates will be removed during the initialisation of the new CoordinateArray instance
        return CoordinateArray(xyzCoords, False, sCoords, AHeadings, AHeadingsFilt)


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
                track limits, in the 2D plane [x, y]. Note that this can be
                provided in the form [x, y, z] but only the [x, y] components
                will be used.
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
        self.xyMidpoint = xyMidpoint[:2]
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
        xyLeft = self.xyMidpoint[:2] + xyVecLeft
        xyRight = self.xyMidpoint[:2] + xyVecRight

        # Generate the gate line
        return shapely.LineString([xyLeft, xyRight])

    def calcIntersection(self,
                         reducedCoordArray: CoordinateArray,
                         key: str = '') -> tuple[float, float]:
        """
        Calculate the gate intersection with a reduced CoordinateArray object.

        Finds the distance from the gate xyMidpoint to the closest intersection
        point, and the distance along the (original) CoordinateArray object of
        the intersection. Then updates the relevant lLimitXY attribute for the
        key provided.

        Args:
            reducedCoordArray: Reduced CoordinateArray object for the local
                region around the gate.
            key: Name of the coordinate array passed in. Automatically updates
                the relevant lLimitXY attribute if the key is one of
                'LimitLeftSoft', 'LimitRightSoft', 'LimitLeftHard' or
                'LimitRightHard'.

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
        lineCoordArray = shapely.LineString(reducedCoordArray.xyzCoords[:, :2])

        # Find the geometry shared between the gate and the reduced coordinate array
        # Note that as the gate is a straight line, the shared geometry with lineCoordArray can only be Shapely Point or Shapely LineString objects,
        # where the LineString objects must be straight lines (unless there are no intersections)
        intersections = shapely.intersection(self.xyLine, lineCoordArray)

        # Check if there was an intersection
        if intersections.is_empty:
            # No intersection between the gate and the reduced coordinate array, set lIntersection to half of the gate max width, and sIntersection to
            # the distance along the coordinate array of the closest point to the gate midpoint
            lIntersection = GATE_MAX_WIDTH / 2
            sIntersection = reducedCoordArray.sCoords[0] + lineCoordArray.project(shapely.Point(self.xyMidpoint))

        else:
            # Gate intersects with the reduced coordinate array, get a list of the intersection geometries (splitting up any MultiPoint geometries)
            intersectionsList = []
            for intersection in intersections.geoms if isinstance(intersections, shapely.GeometryCollection) else [intersections]:
                if isinstance(intersection, shapely.MultiPoint):
                    for point in intersection.geoms:
                        intersectionsList.append(point)
                else:
                    intersectionsList.append(intersection)

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

            # Find the distance along the reduced coordinate array of the closest intersection, and offset it by the distance at the start of the
            # reduced CoordinateArray object to get the distance along the original CoordinateArray object
            sIntersection = lineCoordArray.project(shapely.Point(xyIntersection)) + reducedCoordArray.sCoords[0]

        # Update the relevant lLimitXY attribute
        match key:
            case ('LimitLeftSoft'):
                self.lLimitLeftSoft = lIntersection
            case ('LimitRightSoft'):
                self.lLimitRightSoft = lIntersection
            case ('LimitLeftHard'):
                self.lLimitLeftHard = lIntersection
            case ('LimitRightHard'):
                self.lLimitRightHard = lIntersection

        return lIntersection, sIntersection

    def updateWidths(self,
                     lLeft: float | None = None,
                     lRight: float | None = None) -> None:
        """
        Updates the gate width attributes, then creates a new gate line from the
        new width attributes.

        Args:
            lLeft: Width of the gate to the left of xyMidpoint. If not provided,
                uses the smaller of lLimitLeftSoft + GATE_EXTEND_WIDTH_SOFT and
                lLimitLeftHard + GATE_EXTEND_WIDTH_HARD.
            lRight: Width of the gate to the right of xyMidpoint. If not
                provided, uses the smaller of
                lLimitRightSoft + GATE_EXTEND_WIDTH_SOFT and
                lLimitRightHard + GATE_EXTEND_WIDTH_HARD.

        Raises:
            ValueError 1: Cannot update gate widths: lLeft not specified but
                default value cannot be used as lLimitLeftHard is None.

            ValueError 2: Cannot update gate widths: lRight not specified but
                default value cannot be used as lLimitRightHard is None.
        """
        # Set default values for lLeft and lRight if not provided
        if lLeft is None:
            if self.lLimitLeftSoft is None or self.lLimitLeftHard is None:
                raise ValueError("Cannot update gate widths: lLeft not specified but default value cannot be used "
                                 "as lLimitLeftSoft and/or lLimitLeftHard are None")
            lLeft = min(self.lLimitLeftSoft + GATE_EXTEND_WIDTH_SOFT, self.lLimitLeftHard + GATE_EXTEND_WIDTH_HARD)
        if lRight is None:
            if self.lLimitRightSoft is None or self.lLimitRightHard is None:
                raise ValueError("Cannot update gate widths: lLeft not specified but default value cannot be used "
                                 "as lLimitRightSoft and/or lLimitRightHard are None")
            lRight = min(self.lLimitRightSoft + GATE_EXTEND_WIDTH_SOFT, self.lLimitRightHard + GATE_EXTEND_WIDTH_HARD)

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
        equal. Also updates all the width and distance-to-limit gate attributes
        with the new values (if they exist).

        Raises:
            ValueError: Cannot recalculate gate midpoint: lLimitLeftSoft and/or
                lLimitRightSoft is None.
        """
        # Check that lLimitLeftSoft and lLimitRightSoft both exist
        if self.lLimitLeftSoft is None or self.lLimitRightSoft is None:
            raise ValueError("Cannot recalculate gate midpoint: lLimitLeftSoft and/or lLimitRightSoft is None")

        # Calculate the distance to shift the midpoint (towards the right)
        lShiftRight = self.lLimitRightSoft - self.lLimitLeftSoft

        # Calculate the shifted gate midpoint and update the xyMidpoint attribute
        xyVec = utils.rotateVectorHeading(np.array([lShiftRight, 0]), self.AHeading)
        self.xyMidpoint += xyVec

        # Update the width and distance-to-limit attributes with the shifted midpoint
        self.lLeft += lShiftRight
        self.lRight -= lShiftRight
        self.lLimitLeftSoft += lShiftRight
        self.lLimitRightSoft -= lShiftRight
        if self.lLimitLeftHard is not None:
            self.lLimitLeftHard += lShiftRight
        if self.lLimitRightHard is not None:
            self.lLimitRightHard -= lShiftRight


class Track:
    """
    Defines the track on which a trajectory can be created and optimised.

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
                 BForceTrackGen: bool = False,
                 BClosedTrackOverride: bool | None = None,
                 BDebug: bool = False) -> None:
        """
        Initialises the Track object.

        Tries to load the Track object from the pkl file in the folder specified
        by trackPath. If that fails, falls back to generating the track from the
        csv and JSON files in that folder.

        Files that will be parsed to CoordinateArray objects must be:
            -   A .csv file in the format where each row is a coordinate in
                the form x,y,z.
            -   Named as one of the track limits 'xyzLimitLeftSoft.csv',
                'xyzLimitRightSoft.csv', 'xyzLimitLeftHard.csv', or
                'xyzLimitRightHard.csv'.
            -   Alternatively, named starting with the prefix 'xyzExtra'.

        The trackPath folder must contain the csv files:
            - xyzLimitLeftSoft.csv and/or xyzLimitLeftHard.csv.
            - xyzLimitRightSoft.csv and/or xyzLimitRightHard.csv.

        Files that will be parsed to event Gate objects must:
            -   Be a .json file with the prefix 'eventData'.
            -   Contain the keys:

                -   'type': String identifying the event type
                -   'xyStartLeft': Left coordinate of the event start gate, as a
                    list of floats in the form [xStartLeft, yStartLeft]. May be
                    omitted if the event type is 'StartFinish'.
                -   'xyStartRight': Right coordinate of the event start gate, as a
                    list of floats in the form [xStartRight, yStartRight]. May
                    be omitted if the event type is 'StartFinish'.
                -   'xyFinishLeft': Left coordinate of the event finish gate, as a
                    list of floats in the form [xFinishLeft, yFinishLeft]. May
                    be omitted if the event type is 'StartFinish'.
                -   'xyFinishRight': Right coordinate of the event finish gate, as
                    a list of floats in the form [xFinishRight, yFinishRight].
                    May be omitted if the event type is 'StartFinish'.
                -   'properties': Dictionary of properties for the event, only
                    required if the event requires properties to be defined.

        Args:
            trackPath: File path to the folder containing the .csv and .json
                files required for track generation.
            BForceTrackGen: If true, overrides and forces the track to be
                generated and will overwrite the track .pkl file in the
                specified folder. If false, will use the automatic logic of
                loading from the .pkl file, and falling back to generating the
                track if it fails.
            BClosedTrackOverride: Overrides the automatic logic for determining
                if the track is a closed circuit. If this is None or not
                provided, whether the track is closed or not is determined by
                the maximum distance from the start to finish coordinates of the
                track limit coordinate arrays. This has no effect if the track
                is loaded from a .pkl file.
            BDebug: Whether to show debug plots of the gate creation process
                during track generation.
        """
        if not BForceTrackGen:
            try:
                # Try to load and initialise from the .pkl file
                self.__initFromPkl(trackPath)
                print("Successfully loaded and initialised Track from .pkl file")
            except FileNotFoundError or AttributeError:
                # Initialisation from .pkl failed
                print("Failed to load and initialise Track from .pkl file")
                BForceTrackGen = True

        if BForceTrackGen:
            # Fall back to initialisation from track generation
            print("Initialising track from track generation")
            self.__initFromTrackGen(trackPath, BClosedTrackOverride, BDebug)


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
                           BClosedTrackOverride: bool | None,
                           BDebug: bool = False) -> None:
        """
        Internal method to initialise the Track object by generating the track
        from the .csv and .json files in the specified path.

        By definition, consecutive track gates are not allowed to intersect.
        This influences the large part of the logic in the gate creation loop.

        First parses the track files to a dictionary of CoordinateArray objects,
        and another dictionary of lists of event Gates objects, by event type.
        In the process, automatically detects and sets the BClosedTrack
        attribute (unless overridden).

        Then does the setup before the gate creation loop. The first gate is
        created from the first coordinates of the soft track limits, and its
        intersections with all the CoordinateArray objects are calculated, using
        their corresponding reduced CoordinateArray objects. However, if this
        first gate intersects with any event gates with a similar heading angle,
        it is not used. The final gate (used for stopping gate creation) is also
        created.

        Next, enters the gate creation loop. This loop is exited once the event
        gate to stop gate creation has been reached. In this loop:
            -   The reduced CoordinateArray objects around the next gate are
                found.
            -   The gate placement is optimised first with a root-finding method
                (for equal convergence/divergence of the track limits on either
                side, and equal distances to the soft track limits on either
                side). If the root-finding method fails, then the fallback
                minimisation method is used (with the same broad objective of
                having the gate point in the "forwards" direction with the
                midpoint in the middle of the soft track limits).
            -   From the optimised parameters of the candidate gate placement,
                the candidate gate is created, its intersections with all the
                reduced CoordinateArray objects are calculated, its width is
                updated to follow the GATE_EXTEND_WIDTH_XX constants, and its
                midpoint is recalculated.
            -   The event gates contained in the track segment from the previous
                gate to the candidate gate (which are within the heading angle
                similarity threshold) are found, their intersections with all
                the reduced CoordinateArray objects are calculated, their widths
                are updated to follow the GATE_EXTEND_WIDTH_XX constants, and
                their midpoints are recalculated.
            -   The event gates contained in the track segment are sorted in
                order of ascending distance of the projection of their midpoint
                onto the line from the previous gate midpoint to the candidate
                gate midpoint. If both the start and finish event gates are
                contained in the track segment, they are ordered such that the
                finish gate is before the start gate.
            -   Checks if any of the event gates contained in the track segment
                have the event to stop the gate creation loop. If so, sets the
                flags to stop the gate creation loop.
            -   Checks consecutive event gates intersect with each other but
                don't share the same line. If so, raises an error.
            -   Checks if the last gate in the gates list intersects with the
                first event gate contained in the track segment. If so, removes
                the last gate in the gates list until this is no longer the
                case.
            -   Checks if the last event gate contained by the track segment
                intersects with the candidate gate. If so, marks the candidate
                gate as invalid.
            -   Checks if the candidate gate intersects with the last gate in
                the gates list. If so, marks the candidate gate as invalid.
            -   If the candidate gate is still valid after all the checks,
                adds it to the track gates.
            -   If the flag to stop the gate creation loop is true and the track
                is closed, checks if the first gate intersects with the last
                gate. If so, removes the last gate in the gates list until this
                is no longer the case.

        Once the gate creation loop has finished, postprocesses and validates
        the gates.

        Then creates the track mesh. A SciPy multivariate interpolator is
        created for each gate, using all the coordinates between the previous
        gate and the next gate, and the extrapolated coordinates of the gate
        left and right coordinates.

        Finally, saves the initialised Track object as a pkl file to the
        trackPath folder, and saves a track plot to the same folder, showing:
            -   All coordinate arrays, coloured by track limit side and type
            -   All track gates, coloured by event type and labelled by index
            -   All track gate intersections with the soft and hard track limits
            -   Filled contours of the track elevation

        Args:
            trackPath: File path to the folder containing the .csv and .json
                files required for track generation.
            BClosedTrackOverride: Overrides the automatic logic for determining
                if the track is a closed circuit. If this is None or not
                provided, whether the track is closed or not is determined by
                the maximum distance from the start to finish coordinates of the
                track limit coordinate arrays.
            BDebug: Whether to show debug plots of the gate creation process
                during track generation.

        Raises:
            ValueError1: Both methods for optimisation of new gate placement
                failed, see track plot in {trackPath}.
            ValueError2: Consecutive event gates intersect, see track plot in
                {trackPath}. Possible fixes are spacing out the event gates
                more, reducing GATE_EXTEND_WIDTH_SOFT or GATE_EXTEND_WIDTH_HARD,
                increasing HEADING_ANGLE_THRESHOLD.
            ValueError3: Track is not closed but there are event finish gates
                before their start gates, see track plot in {trackPath}.
            ValueError4: Gate creation stopped before all event gates were added,
                see track plot in {trackPath}. Event gates not added are
                {[f"{gates[i].event.name} (Start {gates[i].event.BStart})"
                for i in indsBadGates]}.
        """
        def __getGateFromCoords(xyLeft: NDArrayFloat1D,
                                xyRight: NDArrayFloat1D,
                                event: Event | None = None,
                                lLimitLeftSoft: float | None = None,
                                lLimitRightSoft: float | None = None,
                                lLimitLeftHard: float | None = None,
                                lLimitRightHard: float | None = None) -> Gate:
            """
            Internal function to initialise a Gate object from its left and
            right coordinates.

            The heading angle generated will be from -pi/4 to 7pi/4. This should
            be checked and modified after the gate is returned to ensure that
            the gate is in the correct rotation.

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
            AHeading = utils.getHeading(xyRight - xyLeft) - (np.pi / 2)
            lHalf = float(np.linalg.norm(xyRight - xyLeft) / 2)

            # Initialise gate
            return Gate(xyMidpoint, AHeading, event, lHalf, lHalf, lLimitLeftSoft, lLimitRightSoft, lLimitLeftHard, lLimitRightHard)

        def __parseTrackFiles() -> tuple[dict[str, CoordinateArray], dict[str, list[Gate]]]:
            """
            Internal function to parse the files in the trackPath folder into
            CoordinateArray and Gate objects, and set the BClosedTrack
            attribute.

            See the docstring for the Track object __init__() method for the
            required file structure within the trackPath folder.

            Parses all csv files with file names matching one of:
                -   'xyzLimitLeftSoft.csv'
                -   'xyzLimitRightSoft.csv'
                -   'xyzLimitLeftHard.csv'
                -   'xyzLimitRightHard.csv'
                -   File name starting with 'xyzExtra' and ending in '.csv'
            into arrays of xyz coordinates. Uses fallbacks for the track limits
            coordinates if not all are provided (uses soft track limits as hard
            track limits if not provided, and vice versa). Determines if the
            track is closed and sets the BClosedTrack attribute. Initialises
            CoordinateArray objects from the arrays of xyz coordinates, and
            makes sure that their initial heading angles are all in the same
            rotation.

            Parses all JSON files starting with 'eventData' to create the
            corresponding event gates. Creates the start and finish gates if
            they are not provided.

            Returns:
                Tuple of (coordArraysDict, eventGatesDict).

                coordArraysDict: Dictionary containing the CoordinateArray
                    objects for all left/right soft/hard track limits, and any
                    extra coordinate arrays provided.

                eventGatesDict: Dictionary containing lists of event Gate
                    objects of each event type.

            Raises:
                ValueError: Event data file {entry.name} has an invalid event
                    type {eventType}: valid event types are {validEventTypes}.
                ValueError: Event data file {entry.name} does not have the key
                    'type' required.
                ValueError: Event data file {entry.name} does not have the
                    {subKey.lower()} coordinate keys 'xyStartLeft' and/or
                    'xyStartRight', and event type is '{eventType}', not
                    'StartFinish'.
                ValueError: {trackPath} must contain at least one of
                    xyzLimit{subKey}Soft.csv or xyzLimit{subKey}Hard.csv.
                ValueError: Event type '{eventType}' has {nStart} start gates
                    but {nFinish} finish gates.
            """
            print("Parsing track files")

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
                        key = entry.name[3:-4] if entry.name in limitFileNames else entry.name[len(extraCoordsFileNamePrefix):-4]
                        xyzCoordsDict[key] = np.genfromtxt(entry, delimiter=',')

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
                                raise ValueError(f"Event data file {entry.name} has an invalid event type {eventType}: valid event types are\n"
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
                                                 f"and/or 'xyStartRight',\nand event type is '{eventType}', not 'StartFinish'")

            # Use fallbacks for left and right track limits coordinates if they don't exist, and automatically determine if the track is closed
            BClosedTrack = True
            xyzCoordsKeys = xyzCoordsDict.keys()
            for subKey in ('Left', 'Right'):
                # Check if fallbacks are needed
                if f'Limit{subKey}Soft' not in xyzCoordsKeys or f'Limit{subKey}Hard' not in xyzCoordsKeys:
                    # Fallbacks needed, use them
                    if f'Limit{subKey}Soft' in xyzCoordsKeys and f'Limit{subKey}Hard' not in xyzCoordsKeys:
                        xyzCoordsDict[f'Limit{subKey}Hard'] = xyzCoordsDict[f'Limit{subKey}Soft']
                    elif f'Limit{subKey}Soft' not in xyzCoordsKeys and f'Limit{subKey}Hard' in xyzCoordsKeys:
                        xyzCoordsDict[f'Limit{subKey}Soft'] = xyzCoordsDict[f'Limit{subKey}Hard']
                    else:
                        raise ValueError(f"{trackPath} must contain at least one of xyzLimit{subKey}Soft.csv or xyzLimit{subKey}Hard.csv")

                # Determine if the track is closed, if all track limits coordinates have their start and finish coordinates within
                # CLOSED_TRACK_THRESHOLD_DISTANCE
                for subKey2 in ('Soft', 'Hard'):
                    xyzCoords = xyzCoordsDict[f'Limit{subKey}{subKey2}']
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
                startGate = __getGateFromCoords(coordArraysDict['LimitLeftSoft'].xyzCoords[0],
                                                coordArraysDict['LimitRightSoft'].xyzCoords[0],
                                                Event('StartFinish', 'StartFinish', True, {}))
            if finishGate is None:
                finishGate = __getGateFromCoords(coordArraysDict['LimitLeftSoft'].xyzCoords[-1],
                                                 coordArraysDict['LimitRightSoft'].xyzCoords[-1],
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
                    raise ValueError(f"Event type '{eventType}' has {nStart} start gates but {nFinish} finish gates")

            print("Finished parsing track files")

            return coordArraysDict, eventGatesDict

        def __getGateFromParams(params: NDArrayFloat1D,
                                prevGate: Gate) -> Gate:
            """
            Internal function to create a Gate object from the optimisation
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
            Internal function to get the objective function vector of the
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
            lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedLimitLeftSoft, 'LimitLeftSoft')
            lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedLimitRightSoft, 'LimitRightSoft')

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
            Internal function to get the objective function value of the candidate
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
            lLimitLeftSoft, _ = candidateGate.calcIntersection(reducedLimitLeftSoft, 'LimitLeftSoft')
            lLimitRightSoft, _ = candidateGate.calcIntersection(reducedLimitRightSoft, 'LimitRightSoft')

            # Calculate the objective function value
            objFunc = lLimitLeftSoft + lLimitRightSoft + abs(lLimitLeftSoft - lLimitRightSoft)

            return objFunc


        def __saveTrackPlot(coordArraysDict: dict[str, CoordinateArray],
                            gates: list[Gate] | np.ndarray[tuple[int], np.dtype[np.object_]],
                            indsBadGates: list[int] | None = None) -> None:
            """
            Internal function to save a plot of the generated track to the
            trackPath folder, primarily for debugging.

            The plot is saved to the trackPath folder, and contains:
                -   All coordinate arrays, coloured by track limit side and type
                -   All track gates, coloured by event type and labelled by
                    index
                -   All track gate intersections with the soft and hard track
                    limits
                -   Filled contours of the track elevation

            Args:
                coordArraysDict: Dictionary containing the CoordinateArray
                    objects for all left/right soft/hard track limits, and any
                    extra coordinate arrays provided.
                gates: List or array of Gate objects to be plotted.
                indsBadGates: List of indexes corresponding to bad gates, which
                    will be plotted as dotted lines.
            """
            # Calculate the axis limits of the plot (limits of the coordinate arrays, plus 1.5 times GATE_EXTEND_WIDTH_HARD)
            lMargin = 1.5 * GATE_EXTEND_WIDTH_HARD
            xMin = min([min(coordArray.xyzCoords[:, 0]) for coordArray in coordArraysDict.values()]) - lMargin
            xMax = max([max(coordArray.xyzCoords[:, 0]) for coordArray in coordArraysDict.values()]) + lMargin
            yMin = min([min(coordArray.xyzCoords[:, 1]) for coordArray in coordArraysDict.values()]) - lMargin
            yMax = max([max(coordArray.xyzCoords[:, 1]) for coordArray in coordArraysDict.values()]) + lMargin

            # Create the figure, main axes and colour bar axes, setting their widths and heights
            axWidth = (xMax - xMin) / TRACK_PLOT_SCALE
            axHeight = (yMax - yMin) / TRACK_PLOT_SCALE
            figWidth = axWidth + TRACK_PLOT_MARGIN * 2
            figHeight = axHeight + TRACK_PLOT_MARGIN * 2
            fig, ax = plt.subplots(1, 1, figsize=(figWidth, figHeight))

            # Plot the coordinate arrays - red/green and bright/dark for left/right soft/hard track limits, or grey if it's an extra coordinate array
            cDict = {'LimitLeftSoft': (1, 0, 0),
                     'LimitRightSoft': (0, 1, 0),
                     'LimitLeftHard': (0.5, 0, 0),
                     'LimitRightHard': (0, 0.5, 0)}
            for key, coordArray in coordArraysDict.items():
                ax.plot(coordArray.xyzCoords[:, 0], coordArray.xyzCoords[:, 1], c=cDict.get(key, (0.5, 0.5, 0.5)), label=key)

            # Plot the track gates, calculate the [x, y] coordinates of the gate's intersections with the track limits, calculate the z coordinates
            # of the points local to the gate
            #  - Colours: black if no event, green if start gate, red if finish gate
            #  - Line style: Dashed if it is a bad gate, solid otherwise
            #  - Annotation: Index of the gate, name and whether it's the start/finish if it's an event gate
            xyIntersectionsDict = {'LimitLeftSoft': np.empty((len(gates), 2)),
                                   'LimitRightSoft': np.empty((len(gates), 2)),
                                   'LimitLeftHard': np.empty((len(gates), 2)),
                                   'LimitRightHard': np.empty((len(gates), 2))}
            annotationSide = 1
            for i, gate in enumerate(gates):
                # Get the colour of the track gate line and text string for the annotation
                c = (0, 0, 0)
                text = str(i)
                if gate.event is not None:
                    text += "\n" + gate.event.name + ("\n(Start)" if gate.event.BStart else "\n(Finish)")
                    if gate.event.type == 'StartFinish':
                        c = (0, 1, 0) if gate.event.BStart else (1, 0, 0)
                if indsBadGates:
                    ls = '--' if i in indsBadGates else '-'
                else:
                    ls = '-'

                # Plot the track gate
                ax.plot(gate.xyLine.xy[0], gate.xyLine.xy[1], c=c, ls=ls)

                # Calculate the [x, y] coordinates of the gate's intersections with the track limits coordinate arrays
                keys = ['LimitLeftSoft', 'LimitRightSoft', 'LimitLeftHard', 'LimitRightHard']
                xyVecLeft = utils.rotateVectorHeading(np.array([-1, 0]), gate.AHeading)
                xyVecRight = utils.rotateVectorHeading(np.array([1, 0]), gate.AHeading)
                xyVecs = [xyVecLeft, xyVecRight, xyVecLeft, xyVecRight]
                lLimits = [gate.lLimitLeftSoft, gate.lLimitRightSoft, gate.lLimitLeftHard, gate.lLimitRightHard]
                for j, key in enumerate(keys):
                    if lLimits[j] is not None:
                        xyIntersectionsDict[key][i] = gate.xyMidpoint + (xyVecs[j] * lLimits[j])
                    else:
                        xyIntersectionsDict[key][i] = np.array([xMin, yMin])

                # Annotate the gate
                xyVecSide = xyVecLeft * gate.lLimitLeftSoft / 2 if annotationSide < 0 else xyVecRight * gate.lLimitRightSoft / 2
                ax.annotate(text, (gate.xyMidpoint[0] + xyVecSide[0], gate.xyMidpoint[1] + xyVecSide[1]), ha='center', va='center')
                annotationSide *= -1

            # Plot the track gate intersections with the track limit coordinate arrays
            keyList = ['LimitLeftSoft', 'LimitRightSoft', 'LimitLeftHard', 'LimitRightHard']
            markerList = ['<', '>', '<', '>']
            fillList = ['none', 'none', 'full', 'full']
            for i, key in enumerate(keyList):
                xy = xyIntersectionsDict[key]
                ax.plot(xy[:, 0], xy[:, 1], c=cDict[key], fillstyle=fillList[i], ls='', marker=markerList[i])

            # Calculate and plot the z coordinates of the track using a colour mesh and contours
            try:
                xCoords = np.arange(np.floor(xMin * TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION) / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION,
                                    np.ceil(xMax * TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION) / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION
                                        + 1 / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION,
                                    1 / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION)
                yCoords = np.arange(np.floor(yMin * TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION) / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION,
                                    np.ceil(yMax * TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION) / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION
                                        + 1 / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION,
                                    1 / TRACK_PLOT_ZMAP_SPATIAL_RESOLUTION)
                zMap = np.full((len(yCoords), len(xCoords)), np.nan)    # matplotlib expects rows as y, columns as x

                # At every gate
                for indGate in range(len(gates)):
                    # Get the indexes of the previous and next gates
                    if self.BClosedTrack:
                        indGatePrev = indGate - 1 if indGate > 0 else len(gates) - 1
                        indGateNext = indGate + 1 if indGate < len(gates) - 1 else 0
                    else:
                        indGatePrev = max(indGate - 1, 0)
                        indGateNext = min(indGate + 1, len(gates) - 1)

                    # Get the left and right coordinates of the current, previous and next gates - in the form [[xLeft, yLeft], [xRight, yRight]]
                    xyGateLine = np.array(gates[indGate].xyLine.xy).T
                    xyGateLinePrev = np.array(gates[indGatePrev].xyLine.xy).T
                    xyGateLineNext = np.array(gates[indGateNext].xyLine.xy).T

                    # Get the coordinates of the virtual gates at the half-step between the gate and the previous/next gate
                    xyLineHalfStepPrev = np.array(((xyGateLine[0] + xyGateLinePrev[0]) / 2, (xyGateLine[1] + xyGateLinePrev[1]) / 2))
                    xyLineHalfStepNext = np.array(((xyGateLine[0] + xyGateLineNext[0]) / 2, (xyGateLine[1] + xyGateLineNext[1]) / 2))

                    # Get the indexes within the rectangular bounding box surrounding the local region around the gate
                    xLocalMin = min(min(xyLineHalfStepPrev[:, 0]), min(xyLineHalfStepNext[:, 0]))
                    xLocalMax = max(max(xyLineHalfStepPrev[:, 0]), max(xyLineHalfStepNext[:, 0]))
                    yLocalMin = min(min(xyLineHalfStepPrev[:, 1]), min(xyLineHalfStepNext[:, 1]))
                    yLocalMax = max(max(xyLineHalfStepPrev[:, 1]), max(xyLineHalfStepNext[:, 1]))
                    xInds = np.where((xLocalMin <= xCoords) & (xCoords <= xLocalMax))[0]
                    yInds = np.where((yLocalMin <= yCoords) & (yCoords <= yLocalMax))[0]
                    if len(xInds) > 0 and len(yInds) > 0:
                        for xInd in xInds:
                            for yInd in yInds:
                                xy = (xCoords[xInd], yCoords[yInd])
                                if np.isnan(zMap[yInd, xInd]) or TRACK_PLOT_OVERWRITE_Z:
                                    # Check if the gate index specified is correct for the specified coordinate
                                    if (utils.getSideOfLine(xy, xyLineHalfStepPrev[0], xyLineHalfStepPrev[1])
                                            <= 0 <= utils.getSideOfLine(xy, xyLineHalfStepNext[0], xyLineHalfStepNext[1])):
                                        # Coordinate xy is between the lines xyLineHalfStepPrev and xyLineHalfStepNext
                                        zMap[yInd, xInd] = self.mesh[indGate](xy[0], xy[1])

                # Plot filled contours - note that matplotlib expects rows as y, columns as x
                indsNotNaN = np.invert(np.isnan(zMap))
                zMin = np.floor(np.min(zMap[indsNotNaN]) / TRACK_PLOT_ZMAP_CONTOUR_INTERVALS) * TRACK_PLOT_ZMAP_CONTOUR_INTERVALS
                zMax = np.ceil(np.max(zMap[indsNotNaN]) / TRACK_PLOT_ZMAP_CONTOUR_INTERVALS) * TRACK_PLOT_ZMAP_CONTOUR_INTERVALS
                contours = ax.contourf(xCoords, yCoords, zMap, cmap='BuPu_r', vmin=zMin, vmax=zMax,
                            levels=np.arange(zMin, zMax + TRACK_PLOT_ZMAP_CONTOUR_INTERVALS, TRACK_PLOT_ZMAP_CONTOUR_INTERVALS))

                # Add colour bar
                caxWidth = TRACK_PLOT_MARGIN / 4
                cax = fig.add_axes(((figWidth - TRACK_PLOT_MARGIN - caxWidth) / figWidth,
                                    TRACK_PLOT_MARGIN / figHeight,
                                    caxWidth / figWidth,
                                    axHeight / figHeight))
                fig.colorbar(contours, cax, location='right', extend='neither')

                # Add the legend - reducing the area that the legend can be placed to avoid interfering with the colour bar
                ax.legend(bbox_to_anchor=(0, 0, 1 - (caxWidth / axWidth), 1))

            except AttributeError:
                # __saveTrackPlot() likely called from a manually raised exception, before the track mesh was created
                # Add the legend - can be placed in any part of the axes
                ax.legend()

            # Add text to explain the notation
            ax.set_title("Track limit coordinates are coloured with red/green for left/right, light/dark for soft/hard\n"
                         "Extra coordinate arrays used for increased track mesh resolution are coloured grey\n"
                         "Track gates are coloured by event type, with markers at their calculated intersections with the track limits\n"
                         "Track gates causing an exception to be raised during track generation are shown with dotted lines",
                         fontsize=plt.rcParams['font.size'])

            # Set the position of the axes, make the axis scales equal, set axis limits
            ax.set_position((TRACK_PLOT_MARGIN / figWidth, TRACK_PLOT_MARGIN / figHeight, axWidth / figWidth, axHeight / figHeight))
            ax.set_aspect('equal')
            ax.set_xlim((xMin, xMax))
            ax.set_ylim((yMin, yMax))

            # Calculate the dpi required and save the track plot
            dpi = TRACK_PLOT_SPATIAL_RESOLUTION * (((xMax - xMin) / axWidth) + ((yMax - yMin) / axHeight))
            fig.savefig(os.path.join(trackPath, TRACK_PLOT_FILENAME), dpi=dpi)
            plt.close(fig)

        ## Parse track files to dictionaries ##
        # Attribute BClosedTrack is also set in the function __parseTrackFiles()
        coordArraysDict, eventGatesDict = __parseTrackFiles()

        ## Setup before gate creation loop ##
        print("Setting up gates")
        # List that will contain Gate objects representing the track gates in order of the direction of travel along the track
        gates: list[Gate] = []
        # Dictionary of lists of the distances along the coordinate arrays of the gate intersections with all coordinate arrays provided, where each
        # index in the list corresponds to the index of the intersecting gate
        sIntersectionsDict: dict[str, list[float]] = {k: [] for k in coordArraysDict.keys()}

        # Create the first gate defined by the first coordinates of the soft track limits, calculating its distances to the soft track limits, and
        # initialising it with the GateCreation event and distances to the soft track limits defined
        lLimitSoft = float(np.linalg.norm(coordArraysDict['LimitLeftSoft'].xyzCoords[0] - coordArraysDict['LimitRightSoft'].xyzCoords[0]) / 2)
        firstGate = __getGateFromCoords(coordArraysDict['LimitLeftSoft'].xyzCoords[0],
                                        coordArraysDict['LimitRightSoft'].xyzCoords[0],
                                        Event('GateCreation', 'GateCreation', True, {}),
                                        lLimitSoft,
                                        lLimitSoft)

        # Update the first gate to the maximum gate width so that intersections with coordinate arrays can be calculated
        firstGate.updateWidths(GATE_MAX_WIDTH / 2, GATE_MAX_WIDTH / 2)


        # Align the first gate's heading angle to be within pi of the average initial filtered heading angle of all coordinate arrays
        AFiltAvg = sum([coordArraysDict[key].AHeadingsFilt[0] for key in coordArraysDict.keys()]) / len(coordArraysDict.keys())
        firstGate.AHeading = utils.wrap(firstGate.AHeading, AFiltAvg - np.pi, AFiltAvg + np.pi)

        # Calculate the gate intersections with all the coordinate arrays
        sDictFirst = {}
        for key, coordArray in coordArraysDict.items():
            if key in ('LimitLeftSoft', 'LimitRightSoft'):
                # Soft track limits intersect at 0 distance along its coordinates, from the definition of the first gate
                sDictFirst[key] = 0

            else:
                # If the coordinate array is not a soft track limit, calculate its reduced CoordinateArray object
                reducedCoordArray = coordArray.getReducedCoordArray(0,
                                                                    -coordArray.sCoords[-1],
                                                                    coordArray.sCoords[-1],
                                                                    coordArray.AHeadingsFilt[0] - REDUCED_HEADING_WINDOW,
                                                                    coordArray.AHeadingsFilt[0] + REDUCED_HEADING_WINDOW)

                # Calculate the gate intersection with the reduced CoordinateArray object, storing the distance along the coordinate array found
                _, sDictFirst[key] = firstGate.calcIntersection(reducedCoordArray, key)

        # Update the first gate's width to follow the GATE_WIDTH_EXTEND constants
        firstGate.updateWidths()

        # Check that the first gate doesn't intersect with any event gates with heading angles within the heading angle threshold
        BValidGate = True
        for eventGatesList in eventGatesDict.values():
            for eventGate in eventGatesList:
                if (firstGate.xyLine.intersects(eventGate.xyLine)
                        and abs(utils.wrap(firstGate.AHeading - eventGate.AHeading, -np.pi, np.pi)) < HEADING_ANGLE_THRESHOLD):
                    BValidGate = False
                    break

        # If the first gate doesn't intersect with any event gates, it's valid, so append the first gate to the gates list and append the distances
        # along the coordinate arrays of its intersections with the coordinate arrays to the relevant lists in sIntersectionsDict
        if BValidGate:
            gates.append(firstGate)
            for key, s in sDictFirst.items():
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
            lLimitSoft = float(np.linalg.norm(coordArraysDict['LimitLeftSoft'].xyzCoords[-1] - coordArraysDict['LimitRightSoft'].xyzCoords[-1]) / 2)
            finalGate = __getGateFromCoords(coordArraysDict['LimitLeftSoft'].xyzCoords[-1],
                                            coordArraysDict['LimitRightSoft'].xyzCoords[-1],
                                            Event('GateCreation', 'GateCreation', False, {}),
                                            lLimitSoft,
                                            lLimitSoft)

        # Add the 'GateCreation' key and value to eventGatesDict
        eventGatesDict['GateCreation'] = [finalGate]

        print("Finished setting up gates")

        ## Gate creation loop ##
        print("Entered gate creation loop")
        prevGate = firstGate
        sDictPrev = sDictFirst
        BStopGateCreation = False
        if BDebug:
            plt.figure(layout='constrained')
        while not BStopGateCreation:
            # Get the reduced CoordinateArray objects and their Shapely LineStrings in the 2D [x, y] plane for each of the coordinate arrays, for the
            # expected region around the gate
            reducedCoordArraysDict: dict[str, CoordinateArray] = {}
            for key, coordArray in coordArraysDict.items():
                sPrev = sDictPrev[key]
                APrev = prevGate.AHeading

                reducedCoordArraysDict[key] = coordArray.getReducedCoordArray(sPrev,
                                                                              sPrev - GATE_STEP_DISTANCE,
                                                                              sPrev + REDUCED_DISTANCE_WINDOW,
                                                                              APrev - REDUCED_HEADING_WINDOW,
                                                                              APrev + REDUCED_HEADING_WINDOW)

            # Optimise the new gate placement using root-finding method
            x0 = np.array([0, 0])
            args = (prevGate, reducedCoordArraysDict['LimitLeftSoft'], reducedCoordArraysDict['LimitRightSoft'])
            sol = scipy.optimize.root(__gateObjFuncRoot, x0, args, method='hybr')

            # Get the optimised candidate gate and calculate its intersections with the soft track limits
            candidateGate = __getGateFromParams(sol.x, prevGate)
            lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedCoordArraysDict['LimitLeftSoft'], 'LimitLeftSoft')
            lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedCoordArraysDict['LimitRightSoft'], 'LimitRightSoft')

            # Check if the root-finding method succeeded
            if sol.success and lLimitLeftSoft < GATE_MAX_WIDTH / 2 or lLimitRightSoft < GATE_MAX_WIDTH / 2:
                print(f"\tRoot finding successfully optimised gate {len(gates) + 1}")
            else:
                # Root-finding failed, fall back to optimisation using minimise, then get the optimised candidate gate and calculate its intersections
                # with the soft track limits
                print(f"\tRoot finding failed for gate {len(gates) + 1}, using fallback minimise method")
                sol = scipy.optimize.minimize(__gateObjFuncMinimize, x0, args, method='BFGS')
                candidateGate = __getGateFromParams(sol.x, prevGate)
                lLimitLeftSoft, sLimitLeftSoft = candidateGate.calcIntersection(reducedCoordArraysDict['LimitLeftSoft'], 'LimitLeftSoft')
                lLimitRightSoft, sLimitRightSoft = candidateGate.calcIntersection(reducedCoordArraysDict['LimitRightSoft'], 'LimitRightSoft')

                # Check if the minimise method failed
                if sol.success and lLimitLeftSoft < GATE_MAX_WIDTH / 2 or lLimitRightSoft < GATE_MAX_WIDTH / 2:
                    print(f"\tFallback minimise method finding successfully optimised gate {len(gates) + 1}")
                else:
                    # Minimise method failed, save the track plot for debugging and raise a ValueError
                    gates.append(candidateGate)
                    indsBadGates = [-1]
                    __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                    raise ValueError(f"Both methods for optimisation of new gate placement failed, see track plot in {trackPath}")

            # Debug plot of reduced coordinate arrays, all gates in the gates list, and the candidate gate
            if BDebug:
                plt.clf()
                for key, coordArray in reducedCoordArraysDict.items():
                    plt.plot(coordArray.xyzCoords[:, 0], coordArray.xyzCoords[:, 1], label=key)
                for gate in gates:
                    plt.plot(gate.xyLine.xy[0], gate.xyLine.xy[1], c=(0, 0, 0))
                plt.plot(candidateGate.xyLine.xy[0], candidateGate.xyLine.xy[1], c=(1, 1, 0))
                lMargin = 1.5 * GATE_EXTEND_WIDTH_HARD
                xMin = min([min(coordArray.xyzCoords[:, 0]) for coordArray in coordArraysDict.values()]) - lMargin
                xMax = max([max(coordArray.xyzCoords[:, 0]) for coordArray in coordArraysDict.values()]) + lMargin
                yMin = min([min(coordArray.xyzCoords[:, 1]) for coordArray in coordArraysDict.values()]) - lMargin
                yMax = max([max(coordArray.xyzCoords[:, 1]) for coordArray in coordArraysDict.values()]) + lMargin
                xAvg = (xMin + xMax) / 2
                yAvg = (yMin + yMax) / 2
                diff = max([xAvg - xMin, yAvg - yMin, xMax - xAvg, yMax - yAvg])
                plt.axis('square')
                plt.xlim(xAvg - diff, xAvg + diff)
                plt.ylim(yAvg - diff, yAvg + diff)
                plt.legend()
                plt.pause(0.1)

            # Calculate the rest of the gate attributes
            sCandidateDict = {}
            for key, reducedCoordArray in reducedCoordArraysDict.items():
                if key == 'LimitLeftSoft':
                    # Already calculated the intersection with the left soft track limits above
                    sCandidateDict[key] = sLimitLeftSoft
                elif key == 'LimitRightSoft':
                    # Already calculated the intersection with the right soft track limits above
                    sCandidateDict[key] = sLimitRightSoft
                else:
                    # Calculate the intersection with the coordinate array and store the distance along the coordinate array of the intersection
                    _, sCandidateDict[key] = candidateGate.calcIntersection(reducedCoordArraysDict[key], key)

            # Update the gate width to follow the GATE_EXTEND_WIDTH_XX constants and recalculate the gate midpoint
            candidateGate.updateWidths()
            candidateGate.recalcMidpoint()

            # Find the event gates contained in the track segment from the previous gate to this candidate gate
            xyLinePrevToCandidateMidpoint = shapely.LineString([prevGate.xyMidpoint, candidateGate.xyMidpoint])
            AHeadingAvg = (prevGate.AHeading + candidateGate.AHeading) / 2
            eventGatesContained: list[Gate] = []
            sEventGatesContained: list[dict[str, float]] = []
            for eventGatesList in eventGatesDict.values():
                for eventGate in eventGatesList:
                    # Calculate the heading angle difference compared to the average heading angle of the previous and candidate gates
                    AHeadingDiff = utils.wrap(eventGate.AHeading - AHeadingAvg, -np.pi, np.pi)

                    # Check if the line from the previous gate midpoint to the candidate gate midpoint intersects with the event gate, and if the
                    # event gate has a similar heading angle to the average heading angle of the previous and candidate gate, and that the previous
                    # gate is not the first gate if the event gate is a GateCreation type
                    if (xyLinePrevToCandidateMidpoint.intersects(eventGate.xyLine)
                            and abs(AHeadingDiff) <= HEADING_ANGLE_THRESHOLD
                            and not (prevGate == firstGate and eventGate.event.type == 'GateCreation')):
                        # Event gate is contained within the track segment from the previous gate to the candidate gate, append it to the
                        # eventGatesContained list
                        print(f"\tEvent gate {eventGate.event.name} of type {eventGate.event.type} "
                              f"({"Start" if eventGate.event.BStart else "Finish"})\n"
                              f"contained in the track segment between the previous and current gates")
                        eventGatesContained.append(eventGate)

                        # Update the event gate width to the maximum gate width to calculate intersections with the coordinate arrays
                        eventGate.updateWidths(GATE_MAX_WIDTH / 2, GATE_MAX_WIDTH / 2)

                        # Calculate the intersections with the coordinate arrays and store the distances along the coordinate arrays of the
                        # intersections
                        sEventGatesContained.append({})
                        for key, reducedCoordArray in reducedCoordArraysDict.items():
                            _, sEventGatesContained[-1][key] = eventGate.calcIntersection(reducedCoordArraysDict[key], key)

                        # Align the event gate's heading angle to be within pi of the average heading angle of the previous and candidate gates
                        eventGate.AHeading = utils.wrap(eventGate.AHeading, AHeadingAvg - np.pi, AHeadingAvg + np.pi)

                        # Update the event gate width to follow the GATE_EXTEND_WIDTH_XX constants and recalculate the event gate midpoint
                        eventGate.updateWidths()
                        eventGate.recalcMidpoint()

            # Sort the eventGatesContained and sEventGatesContained lists in order of ascending distance of the projection of their midpoint onto the
            # line from the previous gate midpoint to the candidate gate midpoint
            if eventGatesContained:
                zipListSorted = sorted(zip(eventGatesContained, sEventGatesContained),
                                       key=lambda x: xyLinePrevToCandidateMidpoint.project(shapely.Point(x[0].xyMidpoint)))
                eventGatesContained = [e for e, _ in zipListSorted]
                sEventGatesContained = [s for _, s in zipListSorted]

            # Check if both the start and finish event gates are in the eventGatesContained list
            indStart = None
            indFinish = None
            for i, eventGate in enumerate(eventGatesContained):
                # Find the indexes of the start and finish gates
                if eventGate.event.type == 'StartFinish' and eventGate.event.BStart:
                    indStart = i
                elif eventGate.event.type == 'StartFinish' and not eventGate.event.BStart:
                    indFinish = i

                if indStart is not None and indFinish is not None:
                    # Both start and finish gates are in the eventGatesContained list, order them such that the finish gate is before the start gate,
                    # in both eventGatesContained and sEventGatesContained
                    eventGatesContained[indStart], eventGatesContained[indFinish] = eventGatesContained[indFinish], eventGatesContained[indStart]
                    sEventGatesContained[indStart], sEventGatesContained[indFinish] = sEventGatesContained[indFinish], sEventGatesContained[indStart]
                    break

            # Create a flag for whether the candidate gate is valid, initialising to true
            BValidGate = True

            if len(eventGatesContained) > 0:
                # Check if any of the event gates contained in the track segment have the event to finish the gate creation loop
                BStopEventGates = [eventGate.event.type == 'GateCreation' and not eventGate.event.BStart for eventGate in eventGatesContained]
                if any(BStopEventGates):
                    # Finish gate creation event gate is contained in the track segment, set the flags to mark the candidate gate as invalid and stop
                    # gate creation
                    BValidGate = False
                    BStopGateCreation = True

                    # Check if the finish gate creation event gate is the last gate in the eventGatesContained list
                    indStopEventGate = BStopEventGates.index(True)
                    if indStopEventGate == len(eventGatesContained):
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

                # Check if consecutive event gates intersect with each other but don't share the same line
                for i in range(len(eventGatesContained) - 1):
                    if (eventGatesContained[i].xyLine.intersects(eventGatesContained[i + 1].xyLine)
                            and (np.linalg.norm(eventGatesContained[i].xyMidpoint - eventGatesContained[i + 1].xyMidpoint)
                                 + abs(eventGatesContained[i].AHeading - eventGatesContained[i + 1].AHeading) > GATE_SIMILARITY_THRESHOLD)):
                        # Intersection found between consecutive event gates that don't share the same line,
                        # save the track plot for debugging and raise a ValueError
                        indsBadGates = [len(gates), len(gates) + i]
                        for eventGate in eventGatesContained:
                            gates.append(eventGate)
                        __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                        raise ValueError(f"Consecutive event gates intersect, see track plot in {trackPath}:\npossible fixes are spacing out the "
                                         f"event gates more,\nreducing GATE_EXTEND_WIDTH_SOFT or GATE_EXTEND_WIDTH_HARD,\n"
                                         f"increasing HEADING_ANGLE_THRESHOLD")

                # Check if the first event gate contained in the track segment intersects with the last gate in the gates list, and if so, remove the
                # last gate in the gates list and repeat until the first event gate no longer intersects with the last gate in the gates list
                BIntersects = True
                while BIntersects and len(gates) > 0:
                    # Check if the first event gate contained in the track segment intersects with the last gate in the gates list,
                    # but doesn't share the same line
                    if (eventGatesContained[0].xyLine.intersects(gates[-1].xyLine)
                            and (np.linalg.norm(eventGatesContained[0].xyMidpoint - gates[-1].xyMidpoint)
                                 + abs(eventGatesContained[0].AHeading - gates[-1].AHeading) > GATE_SIMILARITY_THRESHOLD)):
                        # Intersection found and gates don't share the same line
                        BIntersects = True
                        if gates[-1].event is not None:
                            # Last gate in the gates list is an event gate
                            gates.append(eventGatesContained[0])
                            indsBadGates = [-2, -1]
                            __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                            raise ValueError(f"Consecutive event gates intersect, see track plot in {trackPath}:\npossible fixes are spacing out the "
                                             f"event gates more,\nreducing GATE_EXTEND_WIDTH_SOFT or GATE_EXTEND_WIDTH_HARD,\n"
                                             f"increasing HEADING_ANGLE_THRESHOLD")
                        # Last gate in the gates list is not an event gate, remove it from the gates list and remove the last elements from all the
                        # lists in sIntersectionsDict
                        del gates[-1]
                        for sIntersections in sIntersectionsDict.values():
                            del sIntersections[-1]
                    else:
                        # No intersection found
                        BIntersects = False

                # Check if the last event gate contained in the track segment intersects with the candidate gate
                if eventGatesContained[-1].xyLine.intersects(candidateGate.xyLine):
                    # Intersection found, mark the candidate gate as invalid
                    print(f"\tGate {len(gates) + 1} invalid (intersects with event gate)")
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
                    print(f"\tGate {len(gates) + 1} invalid (intersects with previous gate)")
                    BValidGate = False

            # If the candidate gate is still valid after all the checks, append it and the distances along the coordinate arrays of its intersections
            # with the coordinate arrays to the relevant lists
            if BValidGate:
                gates.append(candidateGate)
                for key in sCandidateDict.keys():
                    sIntersectionsDict[key].append(sCandidateDict[key])
                print(f"\tGate {len(gates) - len(eventGatesContained)} satisfied all checks")

            # If the gate creation finish gate has been reached and the track is closed, check if the first gate intersects with the last gate in the
            # gates list, and if so, remove the last gate in the gates list and repeat until the first gate no longer intersects with the last gate
            # in the gates list
            if BStopGateCreation and self.BClosedTrack:
                BIntersects = True
                while BIntersects and len(gates) > 0:
                    # Check if the first gate in the gates list intersects with the last gate in the gates list, but doesn't the same line
                    if (gates[0].xyLine.intersects(gates[-1].xyLine)
                            and (np.linalg.norm(gates[0].xyMidpoint - gates[-1].xyMidpoint)
                                 + abs(gates[0].AHeading - gates[-1].AHeading) > GATE_SIMILARITY_THRESHOLD)):
                        # Intersection found and gates don't share the same line
                        BIntersects = True
                        if gates[-1].event is not None:
                            if gates[-1].event.type != 'GateCreation':
                                # Last gate in the gates list is an event gate
                                gates.append(eventGatesContained[0])
                                indsBadGates = [-2, -1]
                                __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                                raise ValueError(f"Consecutive event gates intersect, see track plot in {trackPath}:\npossible fixes are spacing out "
                                                 f"the event gates more,\nreducing GATE_EXTEND_WIDTH_SOFT or GATE_EXTEND_WIDTH_HARD,\n"
                                                 f"increasing HEADING_ANGLE_THRESHOLD")
                        # Last gate in the gates list is not an event gate (or it's the event gate to stop gate creation), remove it from the gates
                        # list and remove the last elements from all the lists in sIntersectionsDict
                        del gates[-1]
                        for sIntersections in sIntersectionsDict.values():
                            del sIntersections[-1]
                    else:
                        # No intersection found
                        BIntersects = False

            # Update prevGate to be the candidate gate,
            # and sDictPrev to be the candidate gate's distances along the coordinate arrays of its intersections with the coordinate arrays
            prevGate = candidateGate
            sDictPrev = sCandidateDict

        print("Finished gate creation loop")
        plt.close()

        ## Postprocessing and validation
        print("Validating and postprocessing the gates")
        # Remove the event of the first/last gate in the gates list if they have the event attribute with the GateCreation type
        for i in (0, -1):
            if gates[i].event is not None:
                if gates[i].event.type == 'GateCreation':
                    gates[i].event = None

        # Set the width to the hard track limits to at least the width of the soft track limits (and add the fallbacks in case either weren't set)
        for gate in gates:
            # Left track limit widths
            if gate.lLimitLeftSoft is not None and gate.lLimitLeftHard is not None:
                gate.lLimitLeftHard = max(gate.lLimitLeftSoft, gate.lLimitLeftHard)
            elif gate.lLimitLeftSoft is not None:
                gate.lLimitLeftHard = gate.lLimitLeftSoft
            elif gate.lLimitLeftHard is not None:
                gate.lLimitLeftSoft = gate.lLimitLeftHard
            # Right track limit widths
            if gate.lLimitRightSoft is not None and gate.lLimitRightHard is not None:
                gate.lLimitRightHard = max(gate.lLimitRightSoft, gate.lLimitRightHard)
            elif gate.lLimitRightSoft is not None:
                gate.lLimitRightHard = gate.lLimitRightSoft
            elif gate.lLimitRightHard is not None:
                gate.lLimitRightSoft = gate.lLimitRightHard

        # Find the indexes of the start and finish gates of each event
        indsEventGatesDict = {}     # Dictionary where each key is an event name, storing a list in the form [indEventStart, indEventFinish]
        for i, gate in enumerate(gates):
            if gate.event is not None:
                # Create the key if it doesn't exist
                indsEventGatesDict[gate.event.name] = indsEventGatesDict.get(gate.event.name, [len(gates), -1])
                # Store the event start/finish gate index
                if gate.event.BStart:
                    indsEventGatesDict[gate.event.name][0] = i
                else:
                    indsEventGatesDict[gate.event.name][1] = i

        if self.BClosedTrack:
            # Track is closed, roll the order of the gates so that the finish gate is the last index in the lists
            shift = indsEventGatesDict['StartFinish'][1] + 1

            # First make sure that all the gates that will be rolled have consistent heading angles to avoid discontinuities
            AOffset = round((gates[-1].AHeading - gates[0].AHeading) / (2 * np.pi)) * (2 * np.pi)
            for gate in gates[shift:]:
                gate.AHeading -= AOffset

            # Roll the order of the gates, and roll the order of the lists of the gate's distances along the coordinate arrays of its intersections
            gates = gates[shift:] + gates[:shift]
            for key, s in sIntersectionsDict.items():
                sIntersectionsDict[key] = s[shift:] + s[:shift]

        else:
            # Track is not closed, check if any event start gates are after their finish gate
            indsBadGates = []
            for indsEventGates in indsEventGatesDict.values():
                if indsEventGates[0] > indsBadGates[1]:
                    indsBadGates.append(indsEventGates)
            if indsBadGates:
                __saveTrackPlot(coordArraysDict, gates, indsBadGates)
                raise ValueError(f"Track is not closed but there are event finish gates before their start gates,\nsee track plot in {trackPath}")

        # Check if there are any remaining event gates that haven't been included in the track
        if any(eventGatesDict.values()):
            # There are event gates that haven't been included in the track
            indsBadGates = []
            for eventGatesList in eventGatesDict.values():
                for eventGate in eventGatesList:
                    gates.append(eventGate)
                    indsBadGates.append(len(gates))
            __saveTrackPlot(coordArraysDict, gates, indsBadGates)
            raise ValueError(f"Gate creation stopped before all event gates were added, see track plot in {trackPath}:\nevent gates not added are "
                             f"{[f"{gates[i].event.name} (Start: {gates[i].event.BStart})\n" for i in indsBadGates]}")

        # Set the gates attribute
        self.gates = np.array(gates)
        print("Successfully validated and postprocessed the gates")

        ## Track mesh creation ##
        print("Creating track mesh")
        # Calculate [x, y, z] coordinates at the left and right coordinates of the gates to "artificially" expand the track mesh area and avoid NaNs
        nGates = len(self.gates)
        xyzGatesLeft = np.empty((nGates, 3))
        xyzGatesRight = np.empty((nGates, 3))
        for i, gate in enumerate(self.gates):
            # Get the [x, y] coordinate at the left and right coordinates of the gate
            xyzGatesLeft[i][:2] = np.array(gate.xyLine.xy)[:, 0]
            xyzGatesRight[i][:2] = np.array(gate.xyLine.xy)[:, 1]

            # Linearly extrapolate the z coordinate at the left and right coordinates of the gate from the z coordinates at the hard track limits
            xp = [-gate.lLimitLeftHard, gate.lLimitRightHard]
            fp = [float(np.interp(sIntersectionsDict['LimitLeftHard'][i],
                                  coordArraysDict['LimitLeftHard'].sCoords,
                                  coordArraysDict['LimitLeftHard'].xyzCoords[:, 2])),
                  float(np.interp(sIntersectionsDict['LimitRightHard'][i],
                                  coordArraysDict['LimitRightHard'].sCoords,
                                  coordArraysDict['LimitRightHard'].xyzCoords[:, 2]))]
            xyzGatesLeft[i][2] = utils.linearInterpExtrap(-gate.lLeft, xp, fp)
            xyzGatesRight[i][2] = utils.linearInterpExtrap(gate.lRight, xp, fp)

        # Create the track mesh, which is an array Scipy multivariate interpolators local to the area around their index's gate
        self.mesh = np.empty(nGates, dtype=scipy.interpolate.LinearNDInterpolator)
        for i, gate in enumerate(self.gates):
            # Create an array that will contain all the [x, y, z] coordinates local to the area around the gate
            xyzCoords = np.array([xyzGatesLeft[i], xyzGatesRight[i]])

            # Find the indexes corresponding to the previous and next gate with a different midpoint to the current gate, and add all the left and
            # right [x, y, z] coordinates of the gates contained between them (inclusive)
            iPrev = i
            BFoundPrev = False
            while not BFoundPrev:
                if iPrev == 0:
                    if self.BClosedTrack:
                        iPrev = nGates
                    else:
                        break   # Reached the start of the track
                iPrev -= 1
                xyzCoords = np.vstack((xyzCoords, xyzGatesLeft[iPrev], xyzGatesRight[iPrev]))
                if all(self.gates[iPrev].xyMidpoint != gate.xyMidpoint):
                    BFoundPrev = True
            iNext = i
            BFoundNext = False
            while not BFoundNext:
                if iNext == nGates - 1:
                    if self.BClosedTrack:
                        iNext = -1
                    else:
                        break   # Reached the end of the track
                iNext += 1
                xyzCoords = np.vstack((xyzCoords, xyzGatesLeft[iNext], xyzGatesRight[iNext]))
                if all(self.gates[iNext].xyMidpoint != gate.xyMidpoint):
                    BFoundNext = True

            # Get the [x, y, z] coordinates of all the reduced coordinate arrays (reduced to be between the previous and next gates)
            for key in coordArraysDict.keys():
                coordArray = coordArraysDict[key]
                sRef = sIntersectionsDict[key][i]
                sLower = sIntersectionsDict[key][iPrev]
                sUpper = sIntersectionsDict[key][iNext]
                if self.BClosedTrack:
                    sRef = utils.wrap(sRef, sLower, sLower + coordArray.sCoords[-1])
                    sUpper = utils.wrap(sUpper, sLower, sLower + coordArray.sCoords[-1])

                try:
                    # Heading bounds set for 159 million spirals, should be enough
                    xyzCoords = np.vstack((xyzCoords, coordArray.getReducedCoordArray(sRef, sLower, sUpper, -1e9, 1e9).xyzCoords))
                except IndexError:
                    # No coordinates within the distance bounds, likely this is a hard track limit on the inside of a very tight corner
                    # and is quite far away from the soft track limit - it should not affect the quality of the interpolator
                    pass

            # Create the interpolator for the region around this gate
            self.mesh[i] = scipy.interpolate.LinearNDInterpolator(xyzCoords[:, :2], xyzCoords[:, 2])
        print("Finished creating track mesh")

        ## Saving ##
        print("Saving track object and track plot to", trackPath)
        with open(os.path.join(trackPath, TRACK_PKL_FILENAME), 'wb') as pklFile:
            pkl.dump(self, pklFile, protocol=pkl.HIGHEST_PROTOCOL)
        __saveTrackPlot(coordArraysDict, self.gates)
        print("Finished track generation")

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
        # Calculate the previous and next gate indexes
        if self.BClosedTrack:
            indGatePrev = indGate - 1 if indGate > 0 else len(self.gates) - 1
            indGateNext = indGate + 1 if indGate < len(self.gates) - 1 else 0
        else:
            indGatePrev = max(indGate - 1, 0)
            indGateNext = min(indGate + 1, len(self.gates) - 1)

        # Get the left and right coordinates of the current, previous and next gates - in the form [[xLeft, yLeft], [xRight, yRight]]
        xyGateLine = np.array(self.gates[indGate].xyLine.xy).T
        xyGateLinePrev = np.array(self.gates[indGatePrev].xyLine.xy).T
        xyGateLineNext = np.array(self.gates[indGateNext].xyLine.xy).T

        # Get the coordinates of the virtual gates at the half-step between the gate and the previous/next gate
        xyLineHalfStepPrev = np.array(((xyGateLine[0] + xyGateLinePrev[0]) / 2, (xyGateLine[1] + xyGateLinePrev[1]) / 2))
        xyLineHalfStepNext = np.array(((xyGateLine[0] + xyGateLineNext[0]) / 2, (xyGateLine[1] + xyGateLineNext[1]) / 2))

        # Check if the gate index specified is correct for the specified coordinate
        if utils.getSideOfLine(xy, xyLineHalfStepPrev[0], xyLineHalfStepPrev[1]) > 0 and indGate != indGatePrev:
            # Coordinate is behind the virtual gate at the half-step between the gate and the previous gate, use previous gate's interpolator
            z = self.getTrackZ(xy, indGatePrev, BReturnNaN)
        elif utils.getSideOfLine(xy, xyLineHalfStepNext[0], xyLineHalfStepNext[1]) < 0 and indGate != indGateNext:
            # Coordinate is ahead of the virtual gate at the half-step between the gate and the next gate, use next gate's interpolator
            z = self.getTrackZ(xy, indGateNext, BReturnNaN)
        else:
            # Index is correct for the coordinate
            z = self.mesh[indGate](xy[0], xy[1])

        # Sanitise the z coordinate
        if np.isnan(z) and not BReturnNaN:
            z = 0

        return z

    def getTrackNormal(self,
                       xy_xyz: list[float] | NDArrayFloat1D,
                       indGate: int,
                       nPoints: int = 5) -> NDArrayFloat1D:
        """
        Approximates the normal vector to the track at the specified [x, y] or
        [x, y, z] coordinate by the average cross product of the vectors to
        nPoints coordinates around the specified coordinate.

        If xy_xyz is in the form [x, y, z], then the z coordinate will be used
        as the unperturbed track z coordinate (i.e. skipping the calculation).

        Note that this approximation is used as the SciPy multivariate
        interpolators don't seem to return their gradients.

        Args:
            xy_xyz: Coordinate to calculate the track normal vector at, in the
                form [x, y] or [x, y, z].
            indGate: Index of the closest/most recent gate. This determines
                which local track region the xy coordinate is in.
            nPoints: Number of points (perturbed vectors) to use to approximate
                the track normal. Must be 3 or greater. Higher values give more
                robust and potentially more accurate approximations, but at the
                cost of increased compute time.

        Returns:
            Upwards-facing normal vector to the track at the specified
            coordinate, scaled to a magnitude of 1. If the specified coordinate
            is outside the interpolation region of the track mesh, returns the
            normal pointing directly upwards [0, 0, 1].
        """
        # Get unperturbed [x, y] and z coordinates
        if len(xy_xyz) == 2:
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

        # Calculate perturb vectors, as a 2D list containing lists of the vectors from the specified coordinate to the perturbed coordinate in each
        # direction, where each coordinate is a NumPy array in the form [x, y, z]
        # If nPoints is even, there will be 2 coordinates per direction, otherwise there will be only 1 coordinate per direction - however this
        # structure is used to ensure that cross-products are only taken between vectors with different directions
        xyzPerturbs = []
        if nPoints % 2 == 0:
            # Even number of points, each perturb vector will have a corresponding perturb vector in the opposite direction
            nPoints = int(nPoints / 2)
            APerturbs = np.linspace(0, np.pi * (1 - (1 / nPoints)), nPoints)
            for A in APerturbs:
                xyPerturb = utils.rotateVectorHeading([0, PERTURB_DISTANCE], A)
                xyzList = []
                for direction in (1, -1):
                    zPerturb = self.getTrackZ(xy + (xyPerturb * direction), indGate, BReturnNaN=True)
                    if not np.isnan(zPerturb):
                        xyzList.append(np.append(xyPerturb * direction, zPerturb - z))
                if len(xyzList) > 0:
                    xyzPerturbs.append(xyzList)
        else:
            APerturbs = np.linspace(0, 2 * np.pi * (1 - (1 / nPoints)), nPoints)
            for A in APerturbs:
                xyPerturb = utils.rotateVectorHeading([0, PERTURB_DISTANCE], A)
                zPerturb = self.getTrackZ(xy + xyPerturb, indGate, BReturnNaN=True)
                if not np.isnan(zPerturb):
                    xyzPerturbs.append([np.append(xyPerturb, zPerturb - z)])

        # If there are less than 2 perturbed directions with valid z coordinates, a cross product normal cannot be calculated,
        # so return the failed track normal
        if len(xyzPerturbs) < 2:
            return np.array([0, 0, 1])

        # Calculate cross products for every combination of vectors from the specified coordinate to the perturbed coordinates, scaling them such that
        # their z components are always positive and their magnitudes are 1
        xyzCrossProducts = []
        while len(xyzPerturbs) > 1:
            mainPerturbDirection = xyzPerturbs.pop()
            for xyzMainPerturb in mainPerturbDirection:
                for perturbDirection in xyzPerturbs:
                    for xyzPerturb in perturbDirection:
                        xyzCrossProduct = np.cross(xyzMainPerturb, xyzPerturb)
                        xyzCrossProduct *= np.sign(xyzCrossProduct[2]) / np.linalg.norm(xyzCrossProduct)
                        xyzCrossProducts.append(xyzCrossProduct)
        xyzCrossProducts = np.array(xyzCrossProducts)

        # Calculate the average cross product vector
        xyzTrackNormal = np.empty(3)
        for i in range(3):
            xyzTrackNormal[i] = np.mean(xyzCrossProducts[:, i])

        # Scale the track normal to a magnitude of 1
        xyzTrackNormal /= np.linalg.norm(xyzTrackNormal)

        return xyzTrackNormal

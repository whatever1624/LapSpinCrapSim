# 2 Track

The track module is responsible for defining the track on which a trajectory can be created and optimised.

This includes the `Track` class, as well as the `CoordinateArray`, `Event` and `Gate` classes used to generate and define the track.

# `CoordinateArray` Class

Array of coordinates and its derived attributes.

Used for track generation to store the coordinates defining the track limits and track mesh.

## Attributes

- `xyzCoords`: Coordinates in order of increasing distance along the coordinate array, where each coordinate is in the form [x, y, z]. 2D array of floats.
- `xyLine`: Line formed by the coordinates in the 2D plane [x, y]. Shapely LineString.
- `sCoords`: Cumulative distance along the coordinates, assuming straight lines between coordinates. Starts from 0 if this is the "full" coordinate array, but can start at any value if this is a "reduced" coordinate array. 1D array of floats.
- `AHeadings`: Unwrapped heading angle along the coordinates, in radians, increasing clockwise with 0 corresponding to positive y (north). 1D array of floats.
- `AHeadingsFilt`: Low-pass filtered AHeadings, with cutoff frequency `LP_FILT_SPATIAL_FREQ` and filter order `LP_FILT_ORDER`. 1D array of floats.

## `__init__()`

Initialises the coordinate array and calculates all of its attributes.

Makes sure that the coordinates are closed if the track is closed, removes consecutive duplicate coordinates, calculates the cumulative distance along the coordinates, then calculates the raw and filtered unwrapped heading angle along the coordinates. If all attributes are provided, uses those (without doing any validation).

Note that the initial unfiltered heading angle will be between -pi and pi. This should be checked and corrected if necessary with the method `rotateHeadings()`.

**Arguments:**

- `xyzCoords`: Array of coordinates in order of increasing distance along the track, where each coordinate is in the form [x, y, z].
- `BClosedTrack`: Whether the coordinate array should be treated as closed. A closed coordinate array means that the last coordinate is equal to the first coordinate.
- `sCoords`: If all of `sCoords`, `AHeadings` and `AHeadingsFilt` are provided, overrides the automatically calculated attribute `sCoords`.
- `AHeadings`: If all of `sCoords`, `AHeadings` and `AHeadingsFilt` are provided, overrides the automatically calculated attribute `AHeadings`.
- `AHeadingsFilt`: If all of `sCoords`, `AHeadings` and `AHeadingsFilt` are provided, overrides the automatically calculated attribute `AHeadingsFilt`.

## `rotateHeadings()`

Offsets the attributes `AHeadings` and `AHeadingsFilt` by the angle `theta`.

This should be used to correct the calculated heading angle attributes to ensure that all `CoordinateArray` objects are in the same rotation.

**Arguments:**

- `theta`: Heading angle in radians, positive clockwise, to offset the `AHeadings` and `AHeadingsFilt` attributes.

## `getReducedCoordArray()`

Get a reduced version of the `CoordinateArray` object which only contains the coordinates and attributes local to the region specified.

The reduced coordinates are all the coordinates moving forward and backward from the reference distance `sRef` along the coordinate array, until a distance or heading bound is reached. The coordinate on the bound is then interpolated for both the start and finish points of the reduced coordinate array. The coordinates are ordered in the direction of travel.

**Arguments:**

- `sRef`: Reference distance along the coordinate array. If the track is not closed, must be within the `sLower` and `sUpper` distance bounds.
- `sLower`: Lower distance bound. If the track is closed, can be provided wrapped or unwrapped (with the exception being the bounds would wrap to enclose the entire track, in which case, must be provided unwrapped for the expected behaviour). If the track is not closed, must be <= `sRef`.
- `sUpper`: Lower distance bound. If the track is closed, can be provided wrapped or unwrapped (with the exception being the bounds would wrap to enclose the entire track, in which case, must be provided unwrapped for the expected behaviour). If the track is not closed, must be <= `sRef`.
- `ALower`: Lower unwrapped heading angle bound.
- `AUpper`: Upper unwrapped heading angle bound.

**Returns:**

Reduced coordinate array, as a `CoordinateArray` object.

**Raises:**

- `ValueError` if the heading angle at the reference distance is outside the heading angle bounds.
- `IndexError` if there are no coordinates within the bounds specified.

# `Event` Class

Event information, which is then stored in the corresponding `Gate` object.

The default initialisation of an `Event` object is the event used for automatically generated "normal" track gates.

**Custom event types:**

- `'Auxiliary'`: Manually placed gates to help with tight corners where the gate creation process may skip gates and lose resolution. Does not require the BStart argument.
- `'SpeedLimiter'`: NOT IMPLEMENTED YET
- `'DRS'`: NOT IMPLEMENTED YET
- `'SLM'`: NOT IMPLEMENTED YET

**Internal event types:**

- `None`: "Normal" track gate.
- `'StartFinish'`: Denotes the start and finish lines.
- `'GateCreation'`: Stops the gate creation process once it has reached the end of the lap.

## Attributes

- `eventType`: Event type, see above for the valid event gate types. String or None.
- `BStart`: Whether the gate object storing the event marks the start or finish of the event. Boolean or None.
- `properties`: Dictionary containing the properties specific to the event type required to completely define it. Empty if the event type does not require any additional information to be defined. Dictionary.

## `__init__()`

Initialises the `Event` object.

**Arguments:**

- `eventType`: Event type, see above for the valid event gate types.
- `BStart`: Whether the gate object storing the event marks the start or finish of the event.
- `properties`: Dictionary containing the properties specific to the event.

**Raises:**

- `ValueError`: If the argument `BStart` is None but the event type requires a boolean `BStart` attribute to be defined.
- `ValueError`: If the `properties` argument doesn’t contain all the required keys for the event type

# `Gate` Class

Information about the gate.

The `Gate` object is represented only on the 2D plane [x, y] - otherwise handling intersections with track limits would be practically impossible. Gates are used to define event start and finish locations and define the track limits used to validate generated trajectories.

## Attributes

- `xyLine`: Straight line from the left coordinate to the right coordinate, in the 2D plane [x, y]. Shapely LineString object.
- `xyMidpoint`: Midpoint between the left and right soft track limits, measured along the line of the gate in the 2D plane [x, y]. 1D array of floats.
- `AHeading`: Unwrapped heading angle of the gate, in radians, increasing clockwise with 0 corresponding to positive y (north). This should align with the unwrapped heading angles of the `CoordinateArray` objects. Float.
- `event`: Information about the event starting/ending at this gate. If this is None, then this is a "normal" gate and does not define an event start or finish location. `Event` object or None.
- `lLeft`: Width of the gate to the left of `xyMidpoint`. Float.
- `lRight`: Width of the gate to the right of `xyMidpoint`. Float.
- `lLimitLeftSoft`: Unsigned distance from `xyMidpoint` to the intersection with the left soft track limit. Float or None.
- `lLimitRightSoft`: Unsigned distance from `xyMidpoint` to the intersection with the right soft track limit. Float or None.
- `lLimitLeftHard`: Unsigned distance from `xyMidpoint` to the intersection with the left hard track limit. Float or None.
- `lLimitRightHard`: Unsigned distance from `xyMidpoint` to the intersection with the right hard track limit. Float or None.

## `__init__()`

Initialises the `Gate` object, setting all the attributes provided and generating the gate line.

**Arguments:**

- `xyMidpoint`: Midpoint of the gate between the left and right soft track limits, in the 2D plane [x, y]. Note that this can be provided in the form [x, y, z] but only the [x, y] components
will be used.
- `AHeading`: Unwrapped heading angle of the gate, in radians, increasing clockwise with 0 corresponding to positive y (north). This should align with the unwrapped heading angles of the `CoordinateArray` objects.
- `event`: Information about the event starting/ending at this gate.
- `lLeft`: Width of the gate to the left of `xyMidpoint`.
- `lRight`: Width of the gate to the right of `xyMidpoint`.
- `lLimitLeftSoft`: Unsigned distance from `xyMidpoint` to the intersection with the left soft track limit.
- `lLimitRightSoft`: Unsigned distance from `xyMidpoint` to the intersection with the right soft track limit.
- `lLimitLeftHard`: Unsigned distance from `xyMidpoint` to the intersection with the left hard track limit.
- `lLimitRightHard`: Unsigned distance from `xyMidpoint` to the intersection with the right hard track limit.

## `calcIntersection()`

Calculate the gate intersection with a reduced `CoordinateArray` object.

Finds the distance from the gate `xyMidpoint` to the closest intersection point, and the distance along the (original) `CoordinateArray` object of the intersection. Then updates the relevant `lLimitXY` attribute for the key provided.

**Arguments:**

- `reducedCoordArray`: Reduced `CoordinateArray` object for the local region around the gate.
- `key`: Name of the coordinate array passed in. Automatically updates the relevant `lLimitXY` attribute if the key is one of `'LimitLeftSoft'`, `'LimitRightSoft'`, `'LimitLeftHard'` or `'LimitRightHard'`.

**Returns:**

Tuple of (`lIntersection`, `sIntersection`).

- `lIntersection`: Unsigned distance from the gate midpoint to the intersection between the gate and `reducedCoordArray`. If there is no intersection, this will be `GATE_MAX_WIDTH / 2`.
- `sIntersection`: Distance along `reducedCoordArray` of the intersection with the gate, as calculated from its `sCoords` attribute. If there is no intersection, this will be the distance along `reducedCoordArray` of the closest point to the gate midpoint.

## `updateWidths()`

Updates the gate width attributes, then creates a new gate line from the new width attributes.

**Arguments:**

- `lLeft`: Width of the gate to the left of `xyMidpoint`. If not provided, uses the smaller of `lLimitLeftSoft + GATE_EXTEND_WIDTH_SOFT` and `lLimitLeftHard + GATE_EXTEND_WIDTH_HARD`.
- `lRight`: Width of the gate to the right of `xyMidpoint`. If not provided, uses the smaller of `lLimitRightSoft + GATE_EXTEND_WIDTH_SOFT` and `lLimitRightHard + GATE_EXTEND_WIDTH_HARD`.

**Raises:**

- `ValueError`: If the `lLeft` or `lRight` arguments are not provided, but the default value cannot be calculated as the required distance-to-limit attributes are None.

## `recalcMidpoint()`

Recalculates and moves the gate midpoint to the true midpoint between the left and right soft track limits.

The resulting distances to the left and right soft track limits will be equal. Also updates all the width and distance-to-limit gate attributes with the new values (if they exist).

**Raises:**

- `ValueError`: If any of the `lLimitLeftSoft` or `lLimitRightSoft` are None.

# `Track` Class

Defines the track on which a trajectory can be created and optimised.

## Attributes

- `BClosedTrack`: Whether the track forms a closed circuit. Boolean.
- `gates`: Track gates, with order corresponding to the forwards direction around the track. NumPy array of `Gate` objects.
- `mesh`: Interpolators for the local track z coordinate around each gate. NumPy array of SciPy multivariate interpolators.

## `__init__()`

Initialises the `Track` object.

Tries to load the `Track` object from the pkl file in the folder specified by `trackPath`. If that fails, falls back to generating the track from the csv and tsv files in that folder.

Files that will be parsed to `CoordinateArray` objects must be:

- A csv file (file extension .csv) in the format where each row is a coordinate in the form `x,y,z`, where going down the rows of coordinates means travelling forward along the track. These
files should not have a header.
- Named as one of the track limits `xyzLimitLeftSoft.csv`, `xyzLimitRightSoft.csv`, `xyzLimitLeftHard.csv`, or `xyzLimitRightHard.csv`.
- Alternatively, named starting with the prefix `'xyzExtra'`.

Note that a soft track limit means that the lap is valid as long as 1 tyre is within this limit - think: painted white lines. A hard track limit means that the whole car must be within this limit for the lap to be valid - think: wall, grass, aggressive kerbing.

The file to be parsed for event Gate objects must have the filename 'eventGates.tsv'. This is a file with tab separated values where each line represents a different event gate. Each line must have the column format below. This file should not have a header.

| `xLeft` | `yLeft` | `xRight` | `yRight` | `eventType` | `BStart` | `properties` |
| --- | --- | --- | --- | --- | --- | --- |

The data in each column of the `eventGates.tsv` file is:

- `xLeft`: x coordinate of the left coordinate of the gate.
- `yLeft`: y coordinate of the left coordinate of the gate.
- `xRight`: x coordinate of the right coordinate of the gate.
- `yRight`: y coordinate of the right coordinate of the gate.
- `eventType`: String identifying the event type, see the `Event` class for the allowed event types.
- `BStart`: Boolean identifying whether the event gate marks the start of the event. This can be given as the (case-insensitive) string `true`/`false`, or as binary `1`/`0`.
- `properties`: Dictionary (parsed as JSON format) of properties for the event, see the `Event` class for what properties must be defined for each event type.

The `properties` column is only required if the event type requires properties to be defined.

If only the `xLeft`, `yLeft`, `xRight` and `yRight` columns are provided, the gate is assumed to have the `'Auxiliary'` event type. This event is used to assist the gate creation process in tight corners or around regions with complicated track limits.

Note that the `eventGates.tsv` file is not required. If not provided, it is assumed that there are no custom events, and that the start/finish gates are at the start/finish of the soft track limits.

**Arguments:**

- `trackPath`: File path to the folder containing the csv and tsv files required for track generation.
- `BForceTrackGen`: If true, overrides and forces the track to be generated and will overwrite the track pkl file in the specified folder. If false, will use the automatic logic of loading from the track pkl file, and falling back to generating the track if it fails.
- `BClosedTrackOverride`: Overrides the automatic logic for determining if the track is a closed circuit. If this is None or not provided, whether the track is closed or not is determined by the maximum distance from the start to finish coordinates of the track limit coordinate arrays. This has no effect if the track is loaded from the track pkl file.
- `BDebug`: Whether to show debug plots of the gate creation process during track generation.

**Raises (if initialising by track generation):**

- `ValueError`: If the event data tsv file has an invalid number of tab-separated columns in a given line.
- `ValueError`: If an invalid event type is used in the event data tsv file.
- `ValueError`: If the event data tsv file has an invalid value in the 'BStart' column.
- `ValueError`: If none of the soft track limit or hard track limit coordinate array csv files are provided for one of the left/ right sides.
- `ValueError`: If the number of start and finish event gates for any of the event types are not equal.
- `ValueError`: If both methods of optimisation for the candidate gate placement failed.
- `ValueError`: If consecutive event gates intersect.
- `ValueError`: If there are 2 start gates for the same event type without having a finish gate in between.
- `ValueError`: If there are 2 finish gates for the same event type without having a start gate in between.
- `ValueError`: If there is an event start gate after its finish gate, but the track is not closed.
- `ValueError`: If gate creation stopped before all the event gates were added.

## `getTrackZ()`

Calculates the z coordinate of the track at the specified [x, y] coordinate using the track mesh (local z coordinate interpolators).

Chooses the closest interpolator such that the [x, y] coordinate is within the half-step backwards/forwards to the previous/next gate. If the [x, y] coordinate is outside the interpolation region, will return `0` or `nan`, depending on the argument `BReturnNaN`.

**Arguments:**

- `xy`: Coordinate to calculate the z coordinate of the track at, in the 2D plane [x, y].
- `indGate`: Index of the closest/most recent gate. This determines which local track region the `xy` coordinate is in.
- `BReturnNaN`: Whether to return `nan` if the `xy` coordinate is outside the interpolation region, or to sanitise it and return `0`.

**Returns:**

z coordinate at the specified [x, y] coordinate.

## `getTrackNormal()`

Approximates the normal vector to the track at the specified [x, y] or [x, y, z] coordinate by the average cross product of the vectors to nPoints coordinates around the specified coordinate.

If `xy_xyz` is in the form [x, y, z], then the z coordinate will be used as the unperturbed track z coordinate (i.e. skipping the calculation).

Note that this approximation is used as the SciPy multivariate interpolators don't seem to return their gradients.

**Arguments:**

- `xy_xyz`: Coordinate to calculate the track normal vector at, in the form [x, y] or [x, y, z].
- `indGate`: Index of the closest/most recent gate. This determines which local track region the `xy` coordinate is in.
- `nPoints`: Number of points (perturbed vectors) to use to approximate the track normal. Must be 3 or greater to return a meaningful track normal. Higher values give more robust and potentially more accurate approximations, but at the cost of increased compute time.

**Returns:**

Upwards-facing normal vector to the track at the specified coordinate, scaled to a magnitude of 1. If the specified coordinate
is outside the interpolation region of the track mesh, returns the normal pointing directly upwards [0, 0, 1].
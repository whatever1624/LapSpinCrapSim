# 2 Track

The track module is responsible for defining the track on which a trajectory can be created and optimised - this includes the Track class, as well as the CoordinateArray, Event and Gate classes used to generate and define the track

# CoordinateArray Class

Contains information about the coordinate array and its related functions, used during track generation

## Attributes

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| xyzCoordArray | 2D NumPy array of floats | Coordinate array where each index is an [x, y, z] coordinate, and each subsequent index increases the distance along the track. |
| sCoords | NumPy array of floats | Cumulative distance along the coordinates, calculated assuming straight lines between the coordinates |
| AHeadings | NumPy array of floats | (Unwrapped) heading angle along the coordinates, with 0 corresponding to positive y (north) and increasing clockwise |
| AHeadingsFilt | NumPy array of floats | Low-pass filtered AHeadings with cutoff frequency LP_FILT_SPATIAL_FREQ and order LP_FILT_ORDER, in the spatial domain |
| BClosedTrack | Boolean | True if the coordinate array should be treated as being closed |

## __init__()

Makes sure that the coordArray is closed if BClosedTrack is true (i.e. last coordinate == first coordinate), then calculates the sCoords and AHeadings arrays

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| coordArray | 2D NumPy array of floats | Coordinate array where each index is an [x, y, z] coordinate, and each subsequent index increases the distance along the track. |
| BClosedTrack | Boolean | If true, the coordinate array will be treated as being closed

If false, the coordinate array will be treated as being open |
| BAllowAHeadingNegativeInit | Boolean | If true, will wrap the initial heading angle to the range from -pi to pi

If false, the initial heading angle will be in the range 0 to 2 pi

Note that this will be applied according to the low-pass filtered heading angle, then applied to the unfiltered heading angle

This is intended to ensure that even when the initial heading of the track is close to 0 (or 2 pi radians), AHeadings is consistent between CoordinateArray objects |

## getReducedCoordArray()

Returns a coordinate array object with the arrays of attributes “reduced” to only those within the specified window defined by cumulative distance and heading bounds

Values on the bounds of the window are linearly interpolated

If the track is not closed, and the distance and heading bounds dictate that the reduced coordinate array should extend beyond the defined coordinates, linearly extrapolates from the first/last low-pass filtered heading angle in AHeadings

Used during track generation for gate creation and track mesh creation

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| sLower | Float | Lower bound for the cumulative distances along the coordinate array (i.e. lower bound of sCoords) |
| sUpper | Float | Upper bound for the cumulative distances along the coordinate array (i.e. upper bound of sCoords) |
| ALower | Float | Lower bound for the low-pass filtered cumulative heading (i.e. lower bound of AHeadingsFilt) |
| AUpper | Float | Upper bound for the low-pass filtered cumulative heading (i.e. upper bound of AHeadingsFilt) |

# Event Class

Contains information about the event gate

## Attributes

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| name | String | Name of the event gates pair - note that there must be a start and finish gate for each event name |
| type | String | Event type - see below for the valid event types |
| BStart | Boolean | Boolean whether this is the start gate (if false, means that this is the finish gate for the event) |
| properties | Dictionary | Dictionary of properties specific to the event type |

## Event Types for Custom Events - NOT IMPLEMENTED YET

Event types and their specific properties available for defining custom events

- **DRS** -
- **SLM** -
- **SpeedLimiter** -

## Internal Event Types

These are event types used during track generation, and are not valid event types for defining custom events

- **GateCreation** - First and final gates to mark when to begin and end gate creation for track generation
- **StartFinish** - Start and finish gates for lap timing

# Gate Class

Contains information about the gate and its related functions

The Gate object is represented only on the 2D plane [x, y] - otherwise handling intersections with track limits would be practically impossible

Note that with the gate, the midpoint attribute is not guaranteed to be the midpoint of the gate between the soft track limits due to the gate finding optimisation process, and the possibility of weird track limits making it impossible to make the gate midpoint be technically correct - and this is also why there are both lWidthLimitLeftSoft and lWidthLimitRightSoft variables

## Attributes

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| line | Shapely LineString | Straight line from left coordinate to right coordinate of the gate, only on the 2D plane [x, y] |
| xyMidpoint | NumPy array of floats | Midpoint of the gate between the soft track limits, only on the 2D plane [x, y] |
| AHeading | Float | Heading angle of the gate, with 0 corresponding to positive y (north) and increasing clockwise

This should be the unwrapped heading in the same rotation corresponding to the CoordinateArray objects |
| event | Event object or None | If this is an Event object, then this gate is an event gate (and information about the event is in the Event object)

If this is None, then this gate is a regular gate |
| lLimitLeftSoft | Float or None | Distance from the gate midpoint to the left soft track limit

Positive indicates the gate midpoint is within this track limit

Initialises to None unless otherwise set |
| lLimitRightSoft | Float or None | Distance from the gate midpoint to the right soft track limit

Positive indicates the gate midpoint is within this track limit

Initialises to None unless otherwise set |
| lLimitLeftHard | Float or None | Distance from the gate midpoint to the left hard track limit

Positive indicates the gate midpoint is within this track limit

Initialises to None unless otherwise set |
| lLimitRightHard | Float or None | Distance from the gate midpoint to the right hard track limit

Positive indicates the gate midpoint is within this track limit

Initialises to None unless otherwise set |
| sLimitLeftSoft | Float | Cumulative distance along the left soft track limit coordinate array of the intersection with the gate |
| sLimitRightSoft | Float | Cumulative distance along the right soft track limit coordinate array of the intersection with the gate |
| sLimitLeftHard | Float | Cumulative distance along the left hard track limit coordinate array of the intersection with the gate |
| sLimitRightHard | Float | Cumulative distance along the right hard track limit coordinate array of the intersection with the gate |

## __init__()

Generates the gate line, either from midpoint coordinate and direction vector, or from left and right coordinates

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| xyMidpoint

*Optional, defaults to an empty array

*Initialisation requires either both xyMidpoint and xyDirection, or both xyLeft and xyRight | NumPy array of floats | Midpoint of the gate between the soft track limits, only on the 2D plane [x, y] |
| AHeading

*Optional, defaults to None

*Initialisation requires either both xyMidpoint and xyDirection, or both xyLeft and xyRight | Float or None | Heading angle of the gate, with 0 corresponding to positive y (north) and increasing clockwise

This should be the unwrapped heading in the same rotation corresponding to the CoordinateArray objects |
| xyLeft

*Optional, defaults to an empty array

*Initialisation requires either both xyMidpoint and xyDirection, or both xyLeft and xyRight | NumPy array of floats | Left coordinate of the gate, only on the 2D plane [x, y] |
| xyRight

*Optional, defaults to an empty array

*Initialisation requires either both xyMidpoint and xyDirection, or both xyLeft and xyRight | Numpy array of floats | Right coordinate of the gate, only on the 2D plane [x, y] |
| lLeft

*Optional, defaults to the constant GATE_MAX_WIDTH / 2 | Float | Width of the gate to the left of the midpoint |
| lRight

*Optional, defaults to the constant GATE_MAX_WIDTH / 2 | Float | Width of the gate to the right of the midpoint |
| lLimitLeftSoft

*Optional, defaults to None | Float or None | Distance from the gate midpoint to the left soft track limit

Positive indicates the gate midpoint is within this track limit |
| lLimitRightSoft

*Optional, defaults to None | Float or None | Distance from the gate midpoint to the right soft track limit

Positive indicates the gate midpoint is within this track limit |
| lLimitLeftHard

*Optional, defaults to None | Float or None | Distance from the gate midpoint to the left hard track limit

Positive indicates the gate midpoint is within this track limit |
| lLimitRightHard

*Optional, defaults to None | Float or None | Distance from the gate midpoint to the right hard track limit

Positive indicates the gate midpoint is within this track limit |
| sLimitLeftSoft

*Optional, defaults to 0 | Float | Cumulative distance along the left soft track limit coordinate array of the intersection with the gate |
| sLimitRightSoft

*Optional, defaults to 0 | Float | Cumulative distance along the right soft track limit coordinate array of the intersection with the gate |
| sLimitLeftHard

*Optional, defaults to 0 | Float | Cumulative distance along the left hard track limit coordinate array of the intersection with the gate |
| sLimitRightHard

*Optional, defaults to 0 | Float | Cumulative distance along the right hard track limit coordinate array of the intersection with the gate |

## calcDist()

Function to calculate the gate intersection with the reduced CoordinateArray object provided (reduced meaning local to the region around the gate), using Shapely distance() and project()

Returns the tuple of (lLimitUnsigned, sLimit):

- lLimitUnsigned is the calculated (unsigned) distance from the gate midpoint to the CoordinateArray object (along the gate line)
- sLimit is the calculated cumulative distance along the CoordinateArray object of the intersection with the gate

If no intersection was found, returns lLimitUnsigned as `GATE_MAX_WIDTH / 2` and sLimit as 0

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| reducedLimit | CoordinateArray object | Reduced track limit CoordinateArray object for the local region around the gate |

## calcLimitSofts()

Calculates the gate intersection with the left and right soft track limits using the class function calcDist(), and using the Utils function sideOfLine() to make the distances to the track limit correctly signed

Then updates the attributes with the calculated values for:

- lLimitLeftSoft
- lLimitRightSoft
- sLimitLeftSoft
- sLimitRightSoft

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| reducedLimitLeftSoft | CoordinateArray object | Reduced left soft track limit CoordinateArray object for the local region around the gate |
| reducedLimitRightSoft | CoordinateArray object | Reduced right soft track limit CoordinateArray object for the local region around the gate |

## calcLimitHards()

Calculates the gate intersection with the left and right hard track limits using the class function calcDist(), and using the Utils function sideOfLine() to make the distances to the track limit correctly signed

Then updates the attributes with the calculated values for:

- lLimitLeftHard
- lLimitRightHard
- sLimitLeftHard
- sLimitRightHard

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| reducedLimitLeftHard | CoordinateArray object | Reduced left hard track limit CoordinateArray object for the local region around the gate |
| reducedLimitRightHard | CoordinateArray object | Reduced right hard track limit CoordinateArray object for the local region around the gate |

## updateWidths()

Updates the attributes lLeft and lRight, then calculates the new left and right coordinates of the gate and updates the gate line to be between these new coordinates

Raises an exception if the specified widths would be lower than the widths to the soft or hard limits, if they have been calculated

Also raises an exception if the specified widths would result in a gate wider than GATE_MAX_WIDTH

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| lLeft

*Optional, default to None | Float | New left width of the gate

If None, asserts that lLimitLeftHard exists, then uses `lLimitLeftHard + GATE_EXTEND_WIDTH` |
| lRight

*Optional, default to None | Float | New right width of the gate

If None, asserts that lLimitRightHard exists, then uses `lLimitRightHard + GATE_EXTEND_WIDTH` |

## recalcMidpoint()

Recalculates and moves the gate midpoint along the gate line such that the new lLimitLeftSoft and lLimitRightSoft are equal

Update the attributes with the new values:

- lLeft
- lRight
- lLimitLeftSoft
- lLimitRightSoft
- lLimitLeftHard (if it was not None)
- lLimitRightHard (if it was not None)

Raises an exception if lLimitLeftSoft and lLimitRightSoft are None

# Track Class

Defines the track on which a trajectory can be created and optimised

## Attributes

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| gates | NumPy array of Gate objects | Track gates, with order corresponding to the forwards direction around the track |
| trackMesh | NumPy array of SciPy interpolators | Interpolators for the local track z coordinate around each gate |

## __init__()

Initialises a Track object, either by loading the Track.pkl file in the provided folder, or if that file does not exist/fails to load, by generating the track from the files in the provided folder

See the initFromPkl() and initFromTrackGen() functions below

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| trackPath | String or PathLike | Path to the folder containing the required files to load the Track object from the Track pkl file, or to generate the track if the pkl does not exist |
| BClosedTrackOverride

*Optional, defaults to None | Boolean or None | Overrides the automatic logic for determining if the track is a closed circuit

Defaults to None, in which case whether the track is closed or not is determined automatically by the distances from the start to finish points of the track limit arrays |
| BForceTrackGen

*Optional, defaults to false | Boolean | If true, overrides the automatic logic and forces the track to be generated and will overwrite the Track pkl file in the trackPath

Defaults to false, in which case the Track pkl file will try to be loaded, and only if it doesn’t exist/fails to load will the track be generated |

## __initFromPkl()

Internal function to initialise the Track object by loading the Track pkl file at path pklPath

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| pklPath | String or PathLike | Path to the track pkl file |

## __initFromTrackGen()

Internal function to initialise the Track object by generating it from the data in the trackPath folder (see the in-depth logic below)

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| trackPath | String or PathLike | Path to the folder containing the required files to load the Track object from the Track pkl file, or to generate the track if the pkl does not exist |
| BClosedTrackOverride

*Optional, defaults to None | Boolean or None | Overrides the automatic logic for determining if the track is a closed circuit

Defaults to None, in which case whether the track is closed or not is determined automatically by whether the maximum distance from the start to finish points of the track limit arrays is less than CLOSED_TRACK_THRESHOLD_DISTANCE |

**Expected File Structure in trackPath**

| *Filename* | *Contents* |
| --- | --- |
| xyzLimitLeftSoft.csv | Coordinates forming the left soft track limit, with each row representing a coordinate in the form:
x,y,z

Going down the rows of coordinates means travelling forward along the track limit

The soft track limit means that the lap is valid as long as 1 tyre is within this limit - think: painted white lines |
| xyzLimitRightSoft.csv | Coordinates forming the right soft track limit, with each row representing a coordinate in the form:
x,y,z

Going down the rows of coordinates means travelling forward along the track limit

The soft track limit means that the lap is valid as long as 1 tyre is within this limit - think: painted white lines |
| xyzLimitLeftHard.csv

*Optional, will automatically use the xyzLimitLeftSoft.csv file for the left hard track limit if not provided | Coordinates forming the left hard track limit, with each row representing a coordinate in the form:
x,y,z

Going down the rows of coordinates means travelling forward along the track limit

The hard track limit means that the whole car must be within this limit for the lap to be valid - think: wall, grass, aggressive sausage kerb etc. |
| xyzLimitRightHard.csv

*Optional, will automatically use the xyzLimitRightSoft.csv file for the right hard track limit if not provided | Coordinates forming the right hard track limit, with each row representing a coordinate in the form:
x,y,z

Going down the rows of coordinates means travelling forward along the track limit

The hard track limit means that the whole car must be within this limit for the lap to be valid - think: wall, grass, aggressive sausage kerb etc. |
| xyStartGateOverride.csv

*Optional, will automatically use the first coordinate pair from xyzLimitLeftSoft.csv and xyzLimitRightSoft.csv to create the start gate if not provided | Coordinates to override the location and orientation of the track’s start gate, in the form:
xLeft,yLeft
xRight,yRight

Note that the actual start gate after track generation may not have the same left and right coordinates as defined here, though it will still lie on the same line as defined, with the same direction |
| xyFinishGateOverride.csv

*Optional, will automatically use the first (if the track is closed) or last (if the track is not closed) coordinate pair from xyzLimitLeftSoft.csv and xyzLimitRightSoft.csv to create the finish gate if not provided | Coordinates to override the location and orientation of the track’s finish gate, in the form:
xLeft,yLeft
xRight,yRight

Note that the actual finish gate after track generation may not have the same left and right coordinates as defined here, though it will still lie on the same line as defined, with the same direction |
| eventData_____.json

*Optional, does not create (custom) event gates for the track if not provided

*Can have multiple files beginning with ‘eventData’ and ending with ‘.json’ to allow multiple events | Contains the keys ‘type’, ‘xyStartLeft’, ‘xyStartRight’, ‘xyFinishLeft’, ‘xyFinishRight’ and optionally ‘properties’ which are used to specify and event

‘type’ is a string identifying the type of event (not including ‘StartFinish’ as that is covered by the override files above or automatically determined)

‘xyStartLeft’ is the left coordinate of the event start gate, as a list of floats in the form [xStartLeft, yStartLeft]

’xyStartRight’ is the right coordinate of the event start gate, as a list of floats in the form [xStartRight, yStartRight]

‘xyFinishLeft’ is the left coordinate of the event finish gate, as a list of floats in the form [xFinishLeft, yFinishLeft]

‘xyFinishRight’ is the right coordinate of the event finish gate, as a list of floats in the form [xFinishLeft, yFinishLeft]

‘properties’ is the dictionary of properties for the event and is only needed if the event requires properties to define it

The name used for the event is parsed from the filename (i.e. the unique filename substring represented by ‘_____’)

Note that the actual gates after track generation may not have the same left and right coordinates as defined here, though they will still lie on the same lines as defined, with the same directions |
| xyzExtraCoordsArray_____.csv

*Optional, will only use the coordinates from the available track limits files to create the track mesh if not provided

*Can have multiple files beginning with ‘xyzExtraCoordsArray’ and ending with ‘.csv’ to allow multiple extra coordinate arrays to create a higher resolution track mesh | Coordinates in the same form as the track limits, with each row representing a coordinate in the form:
x,y,z

Going down the rows of coordinates means travelling forward along the track

These are expected to be within the hard track limits (i.e. enclosed by xyzLimitLeftHard and xyzLimitRightHard), though some tolerance is permitted by the gate extending some distance beyond the hard track limits |

### __initGateFromCoords()

Initialise a Gate object (with the minimum initialisation) by processing the left and right coordinates into the midpoint coordinate and heading angle before passing them into the Gate initialisation function

Returns a Gate object

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| xyLeft | NumPy array of floats | Left coordinate of the gate, only on the 2D plane [x, y] |
| xyRight | NumPy array of floats | Right coordinate of the gate, only on the 2D plane [x, y] |
| BAllowNegativeAHeading | Boolean | If true, will wrap the calculated heading angle to the range from -pi to pi

If false, the calculated gate heading angle will be in the range 0 to 2 pi |

### __parseTrackFiles()

Internal function which follows the fallback logic detailed above in the expected file structure of trackPath, to parse the track files into the variables:

- coordinateArraysDict
    - Dictionary containing the CoordinateArray objects for track limits and any extra coordinate arrays provided, with the keys corresponding to their respective file names
    - This will always contain the keys:
        - limitLeftSoft
        - limitRightSoft
        - limitLeftHard
        - limitRightHard
    - Any extra coordinate arrays provided will be stored with the key corresponding to the filename (without the ‘xyz’ on the front and without the file extension)
    - A dictionary for all coordinate arrays is used (over separate variables for each limit) to make the track mesh creation easier in terms of getting all the coordinate arrays
- eventGatesDict
    - Has a key for each (valid) event type, with the corresponding value being a list containing Gate objects with that event type
    - For the ‘StartFinish’ key, its value will be [startGate, finishGate] in that order
    - Note that this dictionary does not contain the ‘GateCreation’ key yet - those event gates will be added in the section below

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| pklPath | String or PathLike | Path to the track pkl file |

### __saveTrackPlot()

Internal function to save a plot of the generated track to the plot file in trackPath, mostly for debugging

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| trackPath | String or PathLike | Path to the folder to save the Track pkl file |
| coordinateArraysDict | Dictionary | Dictionary containing the CoordinateArray objects for track limits and any extra coordinate arrays provided |
| gates | List or NumPy array of Gate objects | Track gates, with order corresponding to the forwards direction around the track |
| badGateIndexes

*Optional, defaults to an empty list | List of integers | Indexes of track gates that caused the exception to be manually thrown

Gates in any of these indexes will be plotted with dotted lines |

Calculates the plot boundaries as the bounds of the track limits coordinate arrays (only need to check the hard limits), plus a margin of 2 * GATE_EXTEND_WIDTH on all sides

Creates a new Matplotlib figure and axes with:

- figsize with the same aspect ratio as the plot boundaries, scaled so the area is equal to TRACK_PLOT_AREA
- layout = ‘constrained’

Sets the x and y limits to the plot boundaries

Sets axis(’equal’) for the correct aspect ratio

Plot track limit coordinate arrays

- Hue/value unique to each limit - dark red for limitLeftHard, light red for limitLeftSoft, light green for limitRightSoft, dark green for limitRightHard
- Saturation decreasing with the z coordinate

Plot extra coordinate arrays (including the 2 generated during track mesh creation)

- Greyscale, with value decreasing with the z coordinate

Plot track gates

- Annotated with the index of the gate, and the name if it is an event gate (with the special case that the start and finish gate have the names ‘Start’ and ‘Finish’ respectively)
- Coloured by event type, with the special cases of:
    - Black if no event
    - Green if start gate
    - Red if finish gate

Scatter plot track gate intersections with the track limits (group by event type and track limit type, then use plot() with the kwarg ls=’’ to be faster)

- Coloured by event type (same as the track gates)
- Markers according to the track limit side:
    - ‘<’ for left track limit
    - ‘>’ for right track limit
- Fill style according to the track limit type:
    - Filled for hard track limit
    - Unfilled for soft track limit

Add text to explain the notation

- Track limit coordinate arrays are coloured with red/green representing left/right, light/dark representing soft/hard, and saturation decreasing with increasing z coordinate
- Extra coordinate arrays used for track mesh creation are greyscale with the value (lightness) decreasing with increasing z coordinate
- Track gates are coloured by event type, with markers representing the calculated intersections with the track limits, with the triangle direction representing left/right, and the fill representing soft/hard
- Track gates that were problematic and raised an exception are shown with dotted lines

Saves the figure with a resolution equivalent to 2 pixels/metre - Find the width (in inches) of the axes from https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels, and use it to calculate the required dpi

### Track Generation Logic

**Read and Parse Track Files to Variables**

Gets coordinateArraysDict and eventGatesDict from parseTrackFiles()

**Setup for Gate Creation**

Create the first gate from the first coordinates in limitLeftSoft and limitRightSoft, with the event specified (`type = 'GateCreation', BStart = True`)

- Run the gate functions to calculate all gate attributes, then update the gate widths to follow the GATE_EXTEND_WIDTH constant
- Check if this gate intersects with startGate - if it intersects, use startGate for the first gate instead and repeat the point above for the startGate

Create the list gates, initialised only containing first gate just created above - this list will contain Gate objects representing the track gates in order of the direction of travel along the track

Then create the final gate - this gate should only span the width between the soft track limits to avoid prematurely stopping gate creation (the gate will be extended and its attributes calculated once it has been reached in the gate creation process)

- If the track is closed, copy the first gate as the stop gate, then modify the gate width to only span the width of the soft track limits and modify the event attribute BStart to false
- If the track is not closed, create the final gate from the final coordinates in limitLeftSoft and limitRightSoft, with the bare minimum initialisation but with the event specified (`type = 'GateCreation', BStart = False`)

Add the ‘GateCreation’ key to eventGatesDict with the value being the list containing only finalGate

**Gate Creation Loop**

Run this loop until the gate creation has reached the last gate, denoted by the flag BStopGateCreation being true

1. Get reduced track limits CoordinateArray objects and Shapely LineStrings (”reduced” as they are for the local region around the gate)
    1. CoordinateArray objects as reducedLimitLeftSoft, reducedLimitRightSoft, reducedLimitLeftHard, and reducedLimitRightHard
        1. Distance window lower bound 0, upper bound to be decided (will be a constant)
        2. Heading window of ±pi radians (180 degrees) from the previous gate heading (will be a constant, subject to change)
    2. Shapely LineStrings computed from the coordinates of the reduced soft track limits CoordinateArray objects, as reducedLimitLeftSoftLS and reducedLimitRightSoftLS, reducedLimitLeftHardLS, and reducedLimitRightHardLS
2. Optimise the new gate placement with SciPy optimise
    1. Find the 2 parameters [psi, theta] to place the gate midpoint in the centre of the track, facing the direction of travel - initial guess of [0, 0]
        1. The midpoint of the new gate will be at distance GATE_STEP_DISTANCE from the midpoint of the previous gate
        2. psi is the anti-clockwise angle in radians from the previous gate direction to place the midpoint of the new gate
        3. theta is the anti-clockwise heading angle offset in radians of the new gate, from psi
    2. Try using root finding first, to solve both residuals below to 0:
        1. `lWidthLimitLeftSoft - lWidthLimitRightSoft` equals 0 when the gate is in the middle of the soft track limits
        2. `ALimitLeftSoft - ALimitRightSoft` equals 0 when the left and right soft track limits are widening/narrowing at the same rate, where these angles are calculated from the AHeadingsFilt attribute of the soft track limits CoordinateArray objects
    3. If the root finding fails, fall back to using local minimisation of the objective function:
        1. `lWidthLimitLeftSoft + lWidthLimitRightSoft + abs(lWidthLimitLeftSoft - lWidthLimitRightSoft)` which is minimised when the gate midpoint is in the centre of the track and the gate widths to the soft track limits are minimised
    4. Raise an exception if the gate creation exploded (optimised parameters cause the gate midpoint to go beyond the bounding box [xMin, xMax, yMin, yMax] of the hard track limits - but run __saveTrackPlot() first with this bad gate added and the badGateIndexes list passed for debugging
3. Create the new gate from the optimised parameters
    1. Runs the gate functions to calculate all gate attributes, then updates the gate widths to follow the GATE_EXTEND_WIDTH constant
4. Create the flag BValidGate for whether this new gate is valid (and should be added to the gates list) or invalid (intersects with the previous or next gate therefore should not be added to the gates list)
    1. Initialise this flag to true
5. Create a list of event gates contained by the track segment from the previous gate to the new gate, sorted by (ascending) distance of their midpoints from the midpoint of the previous gate
    1. Whether an event gate is contained by the track segment is determined by satisfying both conditions:
        1. Dot product between the event gate direction vector and the vector from the midpoint of the previous gate to the midpoint of the new gate exceeds DIRECTION_SIMILARITY_THRESHOLD
        2. Line from the midpoint of the previous gate to the midpoint new gate intersects the event gate line
    2. For each event gate in the list:
        1. Run the gate functions to calculate all gate attributes - calcLimitSofts(), calcLimitHards()
        2. Update the gate widths to follow GATE_EXTEND_WIDTH - updateWidths()
        3. Recalculate the gate midpoint to be centred in the track - recalcMidpoint()
    3. Sort the list in order of (ascending) distance of their midpoints from the midpoint of the previous gate
        1. If the list contains 2 event gates with attributes `type == 'StartFinish'`, sorts such that the gate with `BStart == False` comes before the gate with `BStart == True` (finish gate comes before start gate if they are in the same track segment)
6. If the list of event gates contained by the track segment from the previous gate to the new gate is not empty:
    1. Check if any of the event gates have both attributes `type == 'GateCreation' and BStart == False` - and if so, set the flag BStopGateCreation to true, set the flag BValidGate to false, and remove the event gates in this list following the ‘GateCreation’ finish gate
        1. If there are multiple event gates still in this list, check if the ‘GateCreation’ finish gate (now last in this list) intersects with the 2nd last gate in this list - and if an intersection is detected, remove the ‘GateCreation’ finish gate
    2. Check if consecutive event gates intersect with each other at a single point (multiple event gates can share the same line)
        1. If an intersection is detected, raise an exception (possible fixes are: redefine the event gates to be more spaced out, reduce GATE_EXTEND_WIDTH, increase DIRECTION_SIMILARITY_THRESHOLD) - but run __saveTrackPlot() first with these gates added and the badGateIndexes list passed for debugging
    3. From the first event gate (closest to the previous gate) in the list of event gates contained by this track segment:
        1. Check if the first event gate in this list (closest to the previous gate) intersects with the last gate in the gates list (i.e. the previous gate)
            1. If they intersect, check if the last gate is an event gate (this does not need the check whether the gates share the same line as the event gates will only be “separated” like this if they were on different lines) - and if so, raise an exception (possible fixes are: redefine the event gates to be more spaced out, reduce GATE_EXTEND_WIDTH, increase DIRECTION_SIMILARITY_THRESHOLD) - but run __saveTrackPlot() first with these gates added and the badGateIndexes list passed for debugging
            2. Remove the last gate from the gates list and repeat this step until the first event gate in this list no longer intersects with the last gate in the gates list
    4. For each event gate in this list:
        1. Append the event gate to the gates list
        2. Remove the event gate element from the respective list in eventGatesDict (where the key corresponding to the list is the same as the event type of the event gate)
    5. If BValidGate is still true, check if the last event gate in this list (closest to the new gate) intersects with the new gate - if they intersect, set the flag BValidGate to false
7. If BValidGate is still true, check if the new gate intersects with the previous gate by checking the gate lines with the Shapely intersects() function - If they intersect, set the flag BValidGate to false
8. If BValidGate is still true, append the new gate to the gates list

**Postprocessing and Validation**

Run __saveTrackPlot()

Convert the gate list to a NumPy array

If the track is closed and the gate with `type == 'StartFinish'` and `BStart == False` comes before the gate with event attribute where `type == 'StartFinish'` and `BStart == True` (finish gate before start gate), use NumPy roll() such that the finsh gate is the last index in the gate list

Raise exceptions if any of the conditions below are satisfied:

- eventGatesDict contains non-empty lists (gate creation stopped before all event gates were added, list all the remaining event gates in the exception message)
- Track is not closed and the gate with event attribute where `type == 'StartFinish'` and `BStart == False` comes before the gate with event attribute where `type == 'StartFinish'` and `BStart == True` (track is point-to-point/not closed, but finish gate is before the start gate)

**Track Mesh Creation**

Generate extrapolated gate coordinates - this is necessary to allow better handling of the track mesh outside the bounds of the hard track limits, as SciPy multivariate interpolation doesn’t support extrapolation

1. Create 2 empty NumPy arrays, each of size (length of gates array, 3), which will contain floats
2. For each track gate, get the left and right [x, y, z] coordinates of the gate (where the z coordinates are linearly extrapolated from the z coordinates of the left and right hard limits) and store them in the index of the NumPy arrays corresponding to the index of the track gate
3. Create 2 new CoordinateArray objects from the left and right sets of extrapolated gate coordinates above, and add these to coordinateArraysDict

Create an empty NumPy array the same length of the gates array, which will contain SciPy multivariate interpolator objects (one of the interpolators for unstructured data) - this will be the meshArray representing the track mesh

At each track gate:

1. Get the reduced CoordinateArray objects of all the CoordinateArray objects in coordinateArraysDict
2. Parse the reduced CoordinateArray objects into:
    1. A 2D NumPy array containing all the [x, y] coordinates in the CoordinateArray objects
    2. Another NumPy array containing all the corresponding z coordinates
3. Initialise the SciPy multivariate interpolator with the arguments below, and store it in the index of meshArray corresponding to the index of the track gate
    1. points: 2D NumPy array containing all the [x, y] coordinates parsed above
    2. values: NumPy array containing all the corresponding z coordinates parsed above

**Saving**

Set all class attributes with the values calculated from the track generation

Pickle the Track object using pkl.dump(self) to the track pkl file in trackPath

## calcTrackZ()

Calculates the z coordinate of the track at the specified [x, y] coordinate, using the local track mesh interpolator corresponding to the gate index provided

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| xy | List or NumPy array of floats | Coordinate to calculate the z coordinate of the track at, in the form [x, y] |
| gateIndex | Integer | Index of the gate, which determines which local track mesh interpolator to use to calculate the z coordinate |
| BReturnNaN
*Optional, defaults to false | Boolean | Flag for whether to sanitise the calculated z coordinate

If true, returns 0 instead of NaN if the xy coordinate was outside the interpolation region

If false, returns NaN if the xy coordinate was outside the interpolation region |

## calcTrackNormal()

Calculates the normal vector to the track at the specified [x, y] or [x, y, z] coordinate from forward differencing of the z coordinate

If the forward differencing fails due to the track z coordinate being NaN, returns the failed track normal [0, 0, 1] (directly upwards like a flat track)

Forward differencing is used as the SciPy multivariate interpolators don’t return the gradients

**Arguments**

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| xy_xyz | List or NumPy array of floats | Coordinate to calculate the z coordinate of the track at, in the form [x, y] or [x, y, z]

If the z coordinate is provided, it will skip the calculation of the unpertubed track z coordinate and just use the provided value - so should calculate faster |
| gateIndex | Integer | Index of the gate, which determines which local track mesh interpolator to use to calculate the track normal |

### Calculation Logic

If the z coordinate is not provided, calculates it from calcTrackZ() with the flag BReturnNaN set to true - if the z coordinate is NaN, returns the failed track normal [0, 0, 1]

In each of the x and y directions:

- Perturbs the specified coordinate in the positive direction, then uses calcTrackZ() with the BReturnNaN flag set to true to calculate the track z coordinate at the perturbed location
- If this perturbed location has a NaN track z coordinate, pertubs in the opposite direction - and if it is still NaN, returns the failed track normal [0, 0, 1]
- Calculates the vector from the specified coordinate to this perturbed coordinate, then scales the vector so it has a magnitude of 1, and reverses the vector if it was perturbed in the negative direction

Calculates the track normal as the cross product between the (unit) vectors from the perturbed points (in the order Y cross X such that the normal points in positive z)
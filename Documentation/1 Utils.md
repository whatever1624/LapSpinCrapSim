# 1 Utils

# Type Aliases

A collection of aliases to simplify type hinting.

This is primarily to make type hinting NumPy arrays easier.

# Utils

A collection of helper functions commonly used by the other modules.

## `wrap()`

Wraps value(s) between the lower (inclusive) and upper (exclusive) bounds.

Supports `x` as a NumPy array of values to be wrapped.

**Arguments:**

- `x`: Float or NumPy array of floats to wrap.
- `lowerBound`: Lower bound for the wrapping. This bound is inclusive.
- `upperBound`: Upper bound for the wrapping. This bound is exclusive.

**Returns:**

If the argument x passed in was a float, returns the wrapped float of x.

If the argument x passed in was a NumPy array of floats, returns an array of floats in same shape where each element is wrapped.

## `linearInterpExtrap()`

Linearly interpolate or extrapolate at the point `x` from the closest 2 data points defined by `xp`, `fp`.

Able to compute linear extrapolation unlike NumPy interp(). For computing single points, faster than NumPy interp() if there is a reasonable chance (>15%) that the point will be calculated using the first pair or last pair of data points.

**Arguments:**

- `x`: x coordinate to evaluate the extrapolated function
- `xp`: x coordinates of the function. Must contain 2 or more elements and be monotonically increasing (though not validated explicitly).
- `fp`: Function values corresponding to the x coordinates of `xp`. Must be the same size as `xp`.

**Returns:**

Extrapolated function at point `x`.

## `getHeading()`

Calculates the heading angle(s) of the vector(s), only in the [x, y] plane.

Supports `xy_xyz` as a NumPy array of coordinates.

**Arguments:**

- `xy_xyz`: Vector or NumPy array of vectors, where each vector is in the form [x, y] or [x, y, z].

**Returns:**

Heading angle (if `xy_xyz` was a 1D array) or NumPy array of heading angles (if `xy_xyz` was a 2D array).

The heading angle is from -pi to pi radians, increasing clockwise with 0 corresponding to the direction [0, 1].

**Raises:**

- `ValueError`: If the argument `xy_xyz` doesn’t have a dimension of 1 or 2.

## `resample()`

Resamples the signal at the frequency specified.

**Arguments:**

- `signal`: Signal, can be temporal or spatial.
- `tsSignal`: Temporal or spatial base that the signal was sampled on. Must be monotonically increasing.
- `fResample`: Resample rate of the signal, in Hz (if temporal) or cycles/m (if spatial).
- `BExtrapolate`: Whether to (linearly) extrapolate the signal such that the resampled signal contains the full temporal or spatial range of the original signal. If false, truncates the resampled signal at the closest resampling point to the end of the original signal.

**Returns:**

Tuple of (`signalResampled`, `tsResampled`).

- `signalResampled`: Resampled signal.
- `tsResampled`: New temporal or spatial base of the resampled signal.

## `filt()`

Filters the signal using a Butterworth filter.

**Arguments:**

- `signal`: Signal, can be temporal or spatial, but must be sampled at regular intervals.
- `fSample`: Sample rate of the signal in Hz (if temporal) or cycles/m (if spatial).
- `filtType`: Type of filter, must be one of the strings supported by the `btype` argument for the SciPy `butter()` function - but to simplify, the options are `'low'`, `'high'`, `'bandpass'`, `'bandstop'`.
- `fCutoff`: Cutoff frequency (if low or high-pass filter) or frequencies in the form [fCutoffLow, fCutoffHigh] (if band-pass or band-stop filter). In Hz (if temporal) or cycles/m (if spatial).
- `nOrder`: Order of the filter.

**Returns:**

Filtered signal, as a NumPy array.

## `getIndsWithoutConsecutiveDuplicates()`

Returns the indexes of the array that omit elements that would cause consecutive duplicates.

**Arguments:**

- `arr`: List or array. Data type must be compatible with the NumPy function `diff()`.
- `axis`: Axis along which to check for consecutive duplicates, defaults to
the last axis.

**Returns:**

Array with the indexes that omit elements causing consecutive duplicates along the axis specified.

## `removeConsecutiveDuplicates()`

Returns the array with consecutive duplicates removed along the axis specified.

Effectively a wrapper around `getIndsWithoutConsecutiveDuplicates()`.

**Arguments:**

- `arr`: List or array. Data type must be compatible with the NumPy function `diff()`.
- `axis`: Axis along which to check for consecutive duplicates, defaults to
the last axis.

**Returns:**

Array with the consecutive duplicates removed along the axis specified.

## `rotateVectorHeading()`

Rotates the vector clockwise in the 2D plane [x, y] by theta radians.

Supports both [x, y] and [x, y, z] coordinates as inputs - but does not change the z component of the vector even if provided.

**Arguments:**

- `xy_xyz`: NumPy array in the form [x, y] or [x, y, z] representing the 2D vector.
- `theta`: Angle in radians to rotate the vector, clockwise on the 2D plane [x, y].

**Returns:**

NumPy array representing the rotated vector, in the form [x*, y*] (if provided a 2D vector) or [x*, y* z] (if provided a 3D vector).

## `getSideOfLine()`

Finds which side of the line that the specified point is.

Supports both [x, y] and [x, y, z] coordinates as inputs - but only computes in the 2D [x, y] plane.

**Arguments:**

- `xy_xyzPoint`: Coordinate of the point, in the form [x, y] or [x, y, z].
- `xy_xyzLineStart`: Coordinate of the start of the line, in the form [x, y] or [x, y, z].
- `xy_xyzLineEnd`: Coordinate of the end of the line, in the form [x, y] or [x, y, z].

**Returns:**

0 if all coordinates are collinear (on the same straight line).

>0 if the point is on the right of the line.

< 0 if the point is on the left of the line.

# Optimisation Progress Tracking (OLD)

## Storing Optimisation Progress Data

### `OPT_PROGRESS_DICT`

A global dictionary storing the optimisation progress data

| *Key* | *Type* | *Description* |
| --- | --- | --- |
| `nEvals` | Int | Number of times the evaluation function has been run |
| `EvalResults` | List of floats | Values of the evaluation function corresponding to each time the evaluation function was run |
| `BestResults` | List of floats | Values of the best result so far up to that point corresponding to each time the evaluation function was run |
| `BestInputs` | Array of floats | Input vector to the objective function that gave the best result up to that point |

# Live Plotting (OLD)

## Storing Plot Data

### `PLOTS_DICT`

A global dictionary storing the live plot, its axes, and the dictionaries associated with the active axes

If a plot is removed, the associated dictionary is removed from here also

| *Key* | *Type* | *Description* |
| --- | --- | --- |
| Fig | matplotlib Figure object | The live plot |
| TrackTrajectoryDict | Dictionary or None | Data for the track and trajectory axes (see below) |
| OptProgressDict | Dictionary or None | Data for the optimisation progress axes |
| LapSimProgressDict | Dictionary or None | Data for the lap sim progress axes |

**TrackTrajectoryDict**

| *Key* | *Type* | *Description* |
| --- | --- | --- |
| Track | Track object | Track object currently/to be plotted |
| Trajectory | Trajectory object | Trajectory object currently/to be plotted |
| LeftLines | List of matplotlib Line2D objects | Left side track limits derived from gate data - made by .plot() |
| RightLines | List of matplotlib Line2D objects | Right side track limits derived from gate data - made by .plot() |
| LeftExtendLines | List of matplotlib Line2D objects | Left side extend limits derived from gate data - made by .plot() |
| RightExtendLines | List of matplotlib Line2D objects | Left side track limits derived from gate data - made by .plot() |
| StartLine | List of matplotlib Line2D objects | Start line - made by .plot() |
| FinishLine | List of matplotlib Line2D objects | Finish line - made by .plot() |
| ControlPoints | matplotlib PathCollection object | Control points of the trajectory spline - made by .scatter() |
| TrajectoryLines | List of matplotlib Line2D objects | Trajectory - made by .plot() |
| TrackLimitsLinesList | List of the list of matplotlib Line2D objects | Lines showing the track limits invalidations - each item in this list is made by .plot() |

**OptProgressDict**

| *Key* | *Type* | *Description* |
| --- | --- | --- |
| ProgressLine | List of matplotlib Line2D objects | Plot of *EvalResults* against number of evaluation function calls - made by .plot() |
| BestLine | List of matplotlib Line2D objects | Plot of *BestResults* against number of evaluation function calls - made by .plot() |

**LapSimProgressDict**

not even started on the lap sim lmao

## Plotting

would be nice to see stacked subplots

- top is track and trajectory
- 2nd is optimisation progress (objective function vs number of objective function calls)
- 3rd is maybe live progress of lap sim calculation? although if it runs fast enough then this isn’t necessary

Always maintains order even if a new plot is added - may need to slot the new plot in the middle
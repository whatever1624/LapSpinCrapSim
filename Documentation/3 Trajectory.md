# 3 Trajectory

---

# Trajectory Class

## Overview

**Trajectory Definition**

Racing line defined by control points

Trajectory formed by interpolating between control points using SciPy *make_interp_spline* (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html#scipy.interpolate.make_interp_spline)

(2D) curvature calculated as Menger curvature (https://en.wikipedia.org/wiki/Menger_curvature), made signed by determining if the points curve left (negative) or right (positive)

**Track Limits**

From the track left/right edge coordinates, “gates” are drawn perpendicular to the track centreline

Each gate is represented with a Shapely *LineString* (https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html#shapely.LineString**)**

Checks the number of track limits gates skipped by making a Shapely *LineString* from the discretized trajectory coordinates, iterating through all gates and using the *intersects* function

However the current approach causes problems with gradient descent since the “track limits punishment” function is stepped so gradient descent is bad with it

## Attributes

| *Attribute* | *Type* | *Description* |
| --- | --- | --- |
| *trajType | String | Trajectory type (’Closed Circuit’, ‘Point to Point’, ‘Point to Point with Run Up’, ‘Single Lap’) |
| CP | *2D NumPy array | Control point locations in [x, y] coordinate form |
| spline | SciPy BSpline object | Spline object of the trajectory generated from the control points |
| pStart | Float | Control point value crossing the start line |
| *pFinish | Float | Control point value crossing the finish line |
| sTotal | Float | Total distance of the trajectory from start to finish (excluding the distances before the start line and after the finish line for non-closed circuit trajectories) |
| sDelta | Float | Actual discretization step size used |
| S | NumPy array | (Signed) distances of the trajectory from the start line at each discretized trajectory point (where positive means after the start line and negative means before the start line) |
| P | NumPy array | Control point values corresponding to the *s* array |
| XYZ | 3D NumPy array | Trajectory points in [x, y, z] coordinate form corresponding to the *s* array |
| curvature | NumPy array | (Signed) curvature at each trajectory point corresponding to the *s* array (where positive means left hand corner and negative means right hand corner) |
| valid | NumPy array | Booleans whether the trajectory point is within track limits (True) or not (False) corresponding to the *s* array |
| sInvalid | Float | Length of track limits violated |

## __init__()

Track limits

- if exceeded limits then do a “limits solve recovery” type thing - idk how necessary this is though since track limits need to be exceeded by quite a lot to completely miss a very wide gate

### Inputs

A * indicates the input is optional

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| trajType | String | Trajectory type (’Closed Circuit’, ‘Point to Point’, ‘Point to Point with Run Up’, ‘Single Lap’) |
| track | Track object | Track object that the trajectory is made for - note that this object already includes the track limits considerations of the car’s width |
| CP | 2D list or 2D NumPy array | Control point locations in [x, y] coordinate form |
| sDelta | Float | Desired discretization step distance |
| carWidth | Float | Width of the car at its widest point |
| *margin | Float | Extra width to leave as a margin to *left* and *right* bounds (defaults to 0) |
| *marginExtend | Float | Extra width to leave as a margin to *leftExtend* and *rightExtend* bounds (defaults to 0) |
| *degree | Int | B-spline degree to use for the interpolation between control points (defaults to 3) |

### To Do

- Improve handling of start line crossing if the trajectory doesn’t cross the start gate
    - Maybe find the closest spline sample point to the start gate (Shapely distance function) with a dot product between their 2D xy directions > threshold
- Add sDelta
- ALTERNATIVELY for track limits:
    - Start with startGate’s gateWide LineString
    - Iterate over trajectory points
        - Make a Shapely LineString from that point to the next point
        - Check if it intersects the next gateWide LineString
        - If it intersects, then check whether the intersection point is within track limits or not based on the carWidth and coordinates of the corresponding gate and gateExtend
        - Also calculate the amount the intersection point exceeds track limits (negative to show the extent that it’s legal)
        - Then increment the gate index
    - If the trajectory type means it has a run-up before the startGate, then iterate over the trajectory points from the start line backwards
    - Sum of the distances exceeded is the penalty
- Elevation and everything related to that (slope, camber, warp, changes to curvature calc to be purely lateral curvature)
    - Use linear interpolation to get elevation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator) and if that returns NaN for out of bounds then use nearest neighbour (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html#scipy.interpolate.NearestNDInterpolator)
    - Requires at least track limits left/right edges to have elevation data to feed into the interpolators, but the more points there are the better with this method
    - May need to do a low pass filter on input elevation points though
    - Pass the 2 interpolation objects into the generateTrajectory function as arguments
- Rework gate generation logic so it can use any arbitrary left/right edge coordinate list while still remaining perpendicular across the track and still being at regular intervals
- Support for all trajectory types:
    - Closed circuit (currently assumed)
    - Point to point like a hillclimb - variations of fixed start point or run-up allowed
    - Qualifying/single-lap

### Ideas

- Calculate arrays of unit vectors representing trajectory direction and another array of unit vectors representing track normal? Probably makes it easier to get camber/slope info when I get to doing 3D
- Experiment with using different spline degrees (higher for more smooth spline)
- Allow a Trajectory object to be initialised from a dictionary definition (potentially with a reduced set of attributes)
- Is it possible to make an interactive matplotlib window to make creating the first spline definition faster?
    - Left-click to add control point, right click remove control point + click in a box to say next corner/prev corner?
    - Control point placement just goes to midpoint of the nearest track limits gate to the cursor
    - Candidate new control point plotted in blue, existing control points plotted in green, if the cursor is hovering over an existing control point the plot that one in red
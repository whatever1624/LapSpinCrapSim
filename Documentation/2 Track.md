# 2 Track

---

# Track Class

## Attributes

| *Attribute* | *Type* | *Description* |
| --- | --- | --- |
| xMin | Float | Smallest x coordinate in the *left*, *right*, *leftExtend* or *rightExtend* arrays |
| xMax | Float | Largest x coordinate in the *left*, *right*, *leftExtend* or *rightExtend* arrays |
| yMin | Float | Smallest y coordinate in the *left*, *right*, *leftExtend* or *rightExtend* arrays |
| yMax | Float | Largest y coordinate in the *left*, *right*, *leftExtend* or *rightExtend* arrays |
| zLinInterp | SciPy LinearNDInterpolator object | Linear interpolator for the track z height generated from the provided [x, y, z] coordinates (not intended to be used from outside the class) |
| zNNInterp | SciPy NearestNDInterpolator
object | Nearest neighbour interpolator for the track z height generated from the provided [x, y, z] coordinates (not intended to be used from outside the class) |
| startGateIndex | Int | Index in the gates arrays of the startGate |
| finishGateIndex | Int | Index in the gates arrays of the startGate |
| gates | NumPy array (of Shapely LineString objects) | Gates (only on the xy plane) to get intersections with the trajectory to compute track limits - make these very wide |
| gatesMidpoint | 2D NumPy array | Midpoints of the default gates in [x, y] coordinate form, corresponding to the *gates* array |
| gatesDirection | 2D NumPy array | Direction of the gates in [x, y] coordinate form (normalised to unit length), corresponding to the *gates* array |
| leftWidths | NumPy array | Width from gateMidpoint to the *left* coordinate array, perpendicular to gateDirection, corresponding to the *gates* array |
| rightWidths | NumPy array | Width from gateMidpoint to the *right* coordinate array, perpendicular to gateDirection, corresponding to the *gates* array |
| leftExtendWidths | NumPy array | Width from gateMidpoint to the *leftExtend* coordinate array, perpendicular to gateDirection, corresponding to the *gates* array |
| rightExtendWidths | NumPy array | Width from gateMidpoint to the *rightExtend* coordinate array, perpendicular to gateDirection, corresponding to the *gates* array |
|  |  |  |
| …z height info |  |  |

## __init__()

Creates z height map from all provided [x, y, z] coordinates

- FOR CONVENTIONAL TRACKS - Creates SciPy linear and nearest neighbour interpolators from all available track elevation data points (left, right, leftExtend, rightExtend and maybe extra ones if they’re providedthrough)

isClosed determined by threshold of max(distance from first to last left coordinate, distance from first to last right coordinate) - and used to determine if startGate = finishGate or if they should be distinct (unless otherwise enforced through optional input argument)

Creates start and finish gates

- If provided then use those
- If any of them aren’t provided then use the first coordinates in the left/right arrays for the start gate, and last coordinates in the left/right arrays for the finish gate

Creates gates at regular distance intervals gateStep

- Starts at midpoint of the “first gate” - made from the first coordinates in the left/right arrays
- To get the next gate, move *gateStep* in a given *headingAngle* (save this coordinate as the candidate gateMidpoint) and draw the candidate gate in a *gateDirection* (as a Shapely LineString of some long length)
    - Get the gate intersections with the track left/right coordinate arrays
        - Only consider track limits points close to the candidate gate
            - Example for the left gate, if the last gate intersected at distance *sPrev* along the left coordinate array, then for this new gate only consider the track limits points from distance (*sPrev* + *gateStep* - *cutOff*) to (*sPrev* + *gateStep* + *cutOff*)
            - Make a Shapely LineString from the reduced track left coordinates, and another from the reduced track right coordinates
            - *This process will also have to be done for the leftExtend and rightExtend coordinates - note if it’s not a closed track then if the start point is “behind” the first leftExtend/rightExtend points then just assume left/right as the extend limits
        - Then calculate left gate width by finding the intersection point of the gate LineString and the left LineString and compute the distance from that intersection and the gateMidpoint (use scipy.linalg.norm)
            - Repeat similar for the right gate width
    - SciPy.minimise to solve the gate heading angle and direction
        - Heading angle constrained to ±45 degrees change from previous gate direction, gate direction constrained to ±45 degrees from heading angle (both constraints could probably be narrowed though)
        - Objective function is minimise *leftGateWidth + rightGateWidth + abs(leftGateWidth - rightGateWidth)*
- For each gate created:
    - Check if the vector (Shapely LineString) from the previous gate to this gate intersects with the startGate or finishGate
        - If it does, append the startGate/finishGate to gates first and calculate all the relevant gate things for it
        - And set the startGateIndex/finishGateIndex
    - Check if a Shapely LineString made from the previous gate from *widthLeftExtend* to*widthRightExtend* intersects with a Shapely LineString of the same type for the new gate
        - If it intersects (i.e. a car within track limits could cross the gates out-of-order), don’t add this gate to anything (but still use its midpoint and direction for getting the next gate)
        - If it doesn’t intersect, then:
            - Append this new gate to *gates*
            - Append this new gate’s gateMidpoint to *gatesMidpoint*
            - Calculate this new gate’s normalised direction vector and append it to *gatesDirection*
            - Append this new gate’s leftWidth to *leftWidths*
            - Append this new gate’s rightWidth to *rightWidths*
            - Calculate this new gate’s leftWidthExtend and append it to *leftExtendWidths*
                - To find the distance to the extend limit, iterate along the extend coordinates (make a LineSegment between each coordinate) until there is an intersection with the gate
                - Start the iteration from the most recent LineSegment with an intersection with a gate
                - For the first gate:
                    - Check if the first extend point is “in front” or “behind” the first gate (use the equation from the old curvature function)
                    - If it’s a closed circuit:
                        - Iterate forwards/backwards based on that result
                    - If it’s a point-to-point:
                        - If the first extend point is behind then iterate forwards
                        - If the first extend point is in front then leftWidthExtend = leftWidth
                - *leftWidthExtend = max(leftWidth, leftWidthExtend)
            - Calculate this new gate’s rightWidthExtend and append it to *rightExtendWidths*
                - *rightWidthExtend = max(rightWidth, rightWidthExtend)
- Stops when the line to the new candidate gate intersects the finish gate and the dot product of the gate directions is positive (or > threshold to be tighter)

### Inputs

A * indicates the input is optional

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| left | 2D list or 2D NumPy array | Track limits left edge points in [x, y, z] coordinate form |
| right | 2D list or 2D NumPy array | Track limits right edge points in [x, y, z] coordinate form |
| *leftExtend | 2D list or 2D NumPy array | Track extend left edge points in [x, y, z] coordinate form (defaults to None - if None then sets leftExtend = left) |
| *rightExtend | 2D list or 2D NumPy array | Track extend right edge points in [x, y, z] coordinate form (defaults to None - if None then sets rightExtend = right) |
| *startGate | Shapely LineString object | Start line drawn from left to right (defaults to None - if None then sets using the first left/right track limits coordinates) |
| *finishGate | Shapely LineString object | Finish line drawn from left to right (defaults to None - if None then sets using the first left/right track limits coordinates) |
| *isClosed | Boolean | If the track is closed or not (defaults to None - if None then determines if the track is closed based on the closeness of the first/last left/right coordinates) |
| *gateStep | Float | Distance between gate midpoints (defaults to 10) |
|  |  |  |
|  |  | stuff not implemented below |
| *useRobustZ | Boolean | Use this to force the value useRobustZ flag in the track dictionary output (defaults to None - if None then automatically determines whether this flag should be True or False) |
| eventGates? |  | for DRS, speed limiter etc.? maybe make like 3 event gates inputs to allow up to 3 event types |
| …extra coordinate arrays for better z height interp? |  |  |

Note: *left* and *right* are “artificial” track limits where up to 3 tyres may exceed this limit (e.g. white line), while *leftExtend* and *rightExtend* are “natural” track limits where no tyres can exceed this limit (e.g. walls, grass, gravel, sausage kerbs)

## getZ()

useRobustZ == False:

- Preprocessing In generateTrack(): From all [x, y, z] coordinates from track limits/track extend left/right edges, make a SciPy LinearNDInterpolator object and a SciPy NearestNDInterpolator object
- Pass the [x, y] coordinates into LinearNDInterpolator - if it gives the z value then return it, if it gives NaN then pass the [x, y] coordinates into NearestNDInterpolator
- *This method is only valid for flat tracks - will break for figure-8 tracks with bridges (e.g. Suzuka) and may break for tracks with “cliff separation” if track limits are exceeded (e.g. Monaco)

useRobustZ == True:

- idk
- Maybe have a “lower level” interpolator and “upper level” interpolator and determine which level to use based on direction vector?
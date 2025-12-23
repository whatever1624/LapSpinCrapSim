# 1 Utils

---

# Miscellaneous Functions

### Wrap()

Wraps an input *x* around the lower (inclusive) and upper (exclusive) bounds

Supports NumPy array as input *x* - not sure if it supports that for the bounds but not important

---

# Optimisation Progress

## Storing Optimisation Progress Data

### globalOptProgressDict

A global dictionary storing the optimisation progress data

| *Key* | *Type* | *Description* |
| --- | --- | --- |
| NumEvals | Int | Number of times the evaluation function has been run |
| EvalResults | List of floats | Values of the evaluation function corresponding to each time the evaluation function was run |
| BestResults | List of floats | Values of the best result so far up to that point corresponding to each time the evaluation function was run |

---

# Live Plotting

## Storing Plot Data

### globalPlotsDict

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
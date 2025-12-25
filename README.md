# README

[https://github.com/whatever1624/LapSpinCrapSim](https://github.com/whatever1624/LapSpinCrapSim)

First GitHub repo yippee!!!

This is a solo project which I started at the start of the 2025 F1 summer shutdown as a passion project (and to learn how to use Git but then procrastinated that to weeks after shutdown ended), and have been working on sporadically since then

The goal of this project is to develop a lap sim which includes integrated optimisation of trajectory, setup, energy management etc.

See the Documentation folder of this repo for more information

Expect a lot of refactoring because I have no clue what I’m doing :D

---

# Modules

[1 Utils](1%20Utils%20252e66e2bd3a80fb9319d451e51e4b1c.md)

[2 Track](2%20Track%20251e66e2bd3a80c4b32ecc2949b3cafa.md)

[3 Trajectory](3%20Trajectory%20254e66e2bd3a8036a34cf73c2b42096f.md)

[4 Quasistatic Lap Sim](4%20Quasistatic%20Lap%20Sim%20251e66e2bd3a80da94d2e49af5e89506.md)

[5 Dynamic Post-Processor](5%20Dynamic%20Post-Processor%20251e66e2bd3a802f878cd7d9e741f36e.md)

[6 Optimisation](6%20Optimisation%20254e66e2bd3a80d492daf86206e8709c.md)

---

# Miscellaneous

## To Do

- Clean up everything
    - Also try and use NumPy as much as possible
- Rework the track and trajectory
    - Rename such that limitSoft is white lines, and limitHard is grass/wall
    - Calculate track limit violations in terms of area violated (i.e. area of the polygon enclosed by the violating part of the trajectory and the track limits
    - Reduce the number of functions in the track file - i don’t think it needs to be that big

## **Docs for Packages**

- https://matplotlib.org/stable/api/pyplot_summary.html
*https://matplotlib.org/stable/api/pyplot_summary.html*
- https://numpy.org/doc/stable/reference/index.html
*https://numpy.org/doc/stable/reference/index.html*
- https://shapely.readthedocs.io/en/stable/geometry.html
*https://shapely.readthedocs.io/en/stable/geometry.html*
- https://docs.python.org/3/library/multiprocessing.html#
*https://docs.python.org/3/library/multiprocessing.html#*

## **Useful Links**

- **Matplotlib live plotting** - Use interactive mode, probably store all plot information in a global dictionary (take elements from the interactive mode and scatter plot examples here https://www.geeksforgeeks.org/python/dynamically-updating-plot-in-matplotlib/)
- **If I try using Scipy optimize again -** https://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function
- **If I wanna optimise using genetic algorithm/particle swarm/etc. (trajectory and/or setup) -** https://pymoo.org/index.html
- **Parallelised Scipy optimise minimise with method L-BFGS-B, may be suited for trajectory or energy management optimisation** - https://pypi.org/project/optimparallel/
- **Parallelised Scipy global optimisation with differential_evolution, may be suited for setup optimisation** - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
- **Spline-based trajectory optimisation research paper -** https://arxiv.org/pdf/2309.09186

## Conventions

### Style

**Global Variables and Constants**

- Defined at the top of each module
- All-caps snake case, does not need to follow the variable conventions listed below

**Coordinates**

- In lowercase as in x, y, z - even at the start of a sentence

**Quotation Marks**

- Single quotation marks for keys, substrings for conditionals etc.
- Double quotation marks for full strings like those in print statements

**Functions**

- Type hints and docstrings for all functions
- Newline for each argument - improves readability when using type hints

**Docstrings**

- Largely Google formatted - https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
    
    ```python
    """
    1-line summary of the module/function, terminated with a full stop.
    
    After a line break, write a more detailed description of the module/function if
    necessary. This can include example use cases. Full sentences should end in a full
    stop. Short bullet points can be used without needing to end with a full stop. For
    functions, the section below must be included (unless the function has no arguments,
    no returned values, and no raised errors/exceptions).
    
    Args (IF THE FUNCTION TAKES ARGUMENTS):
    		Function argument 1: Description of the argument. This should contain the data
    				type(s) expected if the type hints are unclear (particularly for lists and
    				arrays). If this runs over a new line, use a tab indent for subsequent lines
    				of the description.
    		Function argument 2: Don't forget the full stop.
    
    Returns (IF THE FUNCTION RETURNS SOMETHING):
    		Describe the return of the function, including the data types(s) expected. If
    		there are multiple returns, this should state "Returns a tuple of (a, b, ...)"
    		and list the values returned similar to the Args section, such as below.
    		a: Description of the returned value. Should contain the data type(s) expected
    		b: Similar things here.
    		This section can also contain logic if the function return changes based on the
    		arguments.
    
    Raises (IF THE FUNCTION RAISES ANYTHING):
    		Error1: Text printed out if the error is raised
    		Error2: Same thing, if there are multiple possible errors that can be raised
    """
    ```
    

### Multiprocessing

To run a single lap sim including dynamic post-processor will be single-threaded

Optimisation can then utilise multiprocessing to run multiple lap sims concurrently

### Variable Conventions

All internal variables and calculations use SI units - unless otherwise stated, assume SI units

Variable name conventions:

- Units not applicable → camelCase
- Units applicable → 1st part describes the unit (lower and/or upper case), 2nd part is the quantity (in PascalCase)

| Character | Unit(s) |
| --- | --- |
| a | Acceleration (m/s^2) |
| A | Angle (rad)
Area (m^2) |
| b |  |
| B | Boolean (-, 1 for True 0 for False) |
| c |  |
| C |  |
| d | Time-derivative modifier (/s, e.g. dmFuel for mass flow rate of fuel, dwYaw for yaw acceleration) |
| D |  |
| e | Efficiency (1, as a ratio) |
| E | Energy (J) |
| f | Frequency (Hz) |
| F | Force (N) |
| g |  |
| G |  |
| h | Relative height (m) |
| H |  |
| i |  |
| I | Current (A) |
| j |  |
| J |  |
| k | Curvature (1/m) |
| K | Spring constant (N/m) |
| l | Length (m) |
| L |  |
| m | Mass (kg) |
| M | Torque/moment (Nm) |
| n | Number (1, e.g. nGear for the gear number) |
| N |  |
| o |  |
| O |  |
| p | Pressure (Pa) |
| P | Power (W) |
| q |  |
| Q |  |
| r | Ratio (1) |
| R | Resistance (ohm) |
| s | Distance/displacement (e.g. sLap) |
| S |  |
| t | Time (s) |
| T | Temperature (K or degC idk which one yet) |
| u |  |
| U |  |
| v | Speed (m/s) |
| V | Volume (m^3)
Voltage (V) |
| w | Angular velocity (rad/s) |
| W |  |
| x | X-coordinate (m) |
| X |  |
| y | Y-coordinate (m) |
| Y |  |
| z | Z-coordinate (m) |
| Z |  |
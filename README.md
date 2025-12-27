# Documentation Homepage

**GitHub Repo:**

[https://github.com/whatever1624/LapSpinCrapSim](https://github.com/whatever1624/LapSpinCrapSim)

**Read the Full Documentation:**

[Documentation Homepage | Notion](https://lapspincrapsim.notion.site/Documentation-Homepage-249e66e2bd3a802daf85e94596e4872e)

First GitHub repo yippee!!!

This is a solo project which I started at the start of the 2025 F1 summer shutdown as a passion project (and to learn how to use Git but then procrastinated that to weeks after shutdown ended), and have been working on sporadically since then

The goal of this project is to develop a lap sim which includes integrated optimisation of trajectory, setup, energy management and even more parameters as this project gets inevitable scope creep

Once I have a working proof-of-concept for all/most of the modules then I will start working with branches, but for now there’s no point when the minimum viable product isn’t even ready

Expect a lot of refactoring because I have no clue what I’m doing :D

# Overview

…this will get filled eventually (i hope)

# Modules

*Note: The links below don’t work (Notion Markdown export), but these are exported to the Documentation folder as Markdown files - though primarily for backup due to bad Markdown formatting of newlines in tables*

[1 Utils](1%20Utils%20252e66e2bd3a80fb9319d451e51e4b1c.md)

[2 Track](2%20Track%202d4e66e2bd3a8026ae52c079d9cee20a.md)

[3 Trajectory OLD](3%20Trajectory%20OLD%20254e66e2bd3a8036a34cf73c2b42096f.md)

[3 Trajectory](3%20Trajectory%202d4e66e2bd3a80c5afb5eccf798ee1c2.md)

[4 Quasistatic Lap Sim](4%20Quasistatic%20Lap%20Sim%20251e66e2bd3a80da94d2e49af5e89506.md)

[5 Dynamic Post-Processor](5%20Dynamic%20Post-Processor%20251e66e2bd3a802f878cd7d9e741f36e.md)

[6 Optimisation](6%20Optimisation%20254e66e2bd3a80d492daf86206e8709c.md)

# Miscellaneous

## To Do

- Clean up everything
    - Also try and use NumPy as much as possible
- Implement the new approach for the track module
- Rework the trajectory module
    - Calculate track limit violations in terms of area violated (i.e. area of the polygon enclosed by the violating part of the trajectory and the track limits

## Required Packages

- https://docs.scipy.org/doc/scipy/
- https://shapely.readthedocs.io/en/stable/
- https://numpy.org/doc/stable/index.html
- https://matplotlib.org/stable/api/pyplot_summary.html

## **Useful Links**

- **Spline-based trajectory optimisation research paper -** https://arxiv.org/abs/2309.09186
- **Python modules for various file formats** - https://opendata.stackexchange.com/questions/1208/a-python-guide-for-open-data-file-formats
- **Matplotlib live plotting** - Use interactive mode, probably store all plot information in a global dictionary (take elements from the interactive mode and scatter plot examples here https://www.geeksforgeeks.org/python/dynamically-updating-plot-in-matplotlib/)
- **SciPy optimize input structure -** https://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function
- **Optimisation using genetic algorithm, particle swarm etc. -** https://pymoo.org/index.html
- **Parallelised SciPy optimise minimise with method L-BFGS-B, may be suited for trajectory or energy management optimisation** - https://pypi.org/project/optimparallel/
- **Parallelised SciPy global optimisation with differential_evolution, may be suited for setup optimisation** - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution
- **Python multiprocessing** - https://docs.python.org/3/library/multiprocessing.html

## Conventions

### Style

**Line Length**

- For docstrings, 80 characters or less
- For code and comments, free to use as much as necessary - though recommended limit of 160 so that 1 line fits on my 14-inch laptop

**Global Variables and Constants**

- Defined at the top of each module
- UPPER_CASE, does not need to follow the variable conventions listed below

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
- Docstring line length must be 80 characters or less
    
    ```python
    """
    Short 1 or 2 sentence summary of the module/function, terminated with a full stop.
    
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
    		
    		a: Description of the returned value. Should contain the data type(s) expected.
    		Do not tab-indent subsequent lines here.
    		
    		b: Similar things here.
    		
    		This section can also contain logic if the function return changes based on the
    		arguments. Note that blank lines are necessary for the documentation tooltip to
    		render line breaks properly.
    
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

Dictionary keys use PascalCase unless units are applicable, in which case it follows the convention below

Variable name conventions:

- Units not applicable → camelCase
- Units applicable → 1st part describes the unit (lower and/or upper case), 2nd part is the quantity (in PascalCase)

| **Character** | **Unit(s)** | **Sign Convention for Positive** |
| --- | --- | --- |
| a | Acceleration (m/s^2) | Forwards |
| A | Angle (rad)

Area (m^2) | Anti-clockwise, unless heading angle in which case clockwise
N/A |
| b |  |  |
| B | Boolean (-, 1 for True 0 for False) | N/A |
| c |  |  |
| C |  |  |
| d | Time-derivative modifier (/s, e.g. dmFuel for mass flow rate of fuel) | Increasing with time |
| D |  |  |
| e | Efficiency (1, as a ratio) | N/A |
| E | Energy (J) | N/A |
| f | Frequency (Hz) | N/A |
| F | Force (N) | Depends |
| g |  |  |
| G |  |  |
| h | Relative height (m) | Higher relative height |
| H |  |  |
| i |  |  |
| I | Current (A) | N/A |
| j |  |  |
| J |  |  |
| k | Curvature (1/m) | Curving to the left |
| K | Spring constant (N/m) | N/A |
| l | Length (m) | N/A |
| L |  |  |
| m | Mass (kg) | N/A |
| M | Torque/moment (Nm) | Follow right-hand rule |
| n | Number (1, e.g. nGear for the gear number) | N/A |
| N |  |  |
| o |  |  |
| O |  |  |
| p | Pressure (Pa) | N/A |
| P | Power (W) | N/A |
| q |  |  |
| Q |  |  |
| r | Ratio (1) | N/A |
| R | Resistance (ohm) | N/A |
| s | Distance/displacement (e.g. sLap) | Forwards |
| S |  |  |
| t | Time (s) | Forwards |
| T | Temperature (K or degC idk which one yet) | N/A |
| u |  |  |
| U |  |  |
| v | Speed (m/s) | N/A |
| V | Volume (m^3)
Voltage (V) | N/A
N/A |
| w | Angular velocity (rad/s) | Anti-clockwise |
| W |  |  |
| x | X-coordinate (m) | Forwards in car coordinates, east in track coordinates |
| X |  |  |
| y | Y-coordinate (m) | Left in car coordinates, north in track coordinates |
| Y |  |  |
| z | Z-coordinate (m) | Up |
| Z |  |  |
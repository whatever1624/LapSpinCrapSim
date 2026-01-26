# Documentation Homepage

**GitHub Repo:**

[https://github.com/whatever1624/LapSpinCrapSim](https://github.com/whatever1624/LapSpinCrapSim)

**Read the Full Documentation:**

[Documentation Homepage | Notion](https://lapspincrapsim.notion.site/Documentation-Homepage-249e66e2bd3a802daf85e94596e4872e)

First GitHub repo yippee!!!

This is a solo project which I started at the start of the 2025 F1 summer shutdown as a passion project (also to learn how to use Git but that got procrastinated to weeks after shutdown), and I’ve been working on it sporadically since then.

The goal of this project is to be able to optimise trajectory, setup, energy management and other parameters (added through inevitable scope creep), using a quasistatic lap sim as the base.

Once I have a working proof-of-concept for all/most of the modules then I will start working with branches, but for now there’s no point when the minimum viable product isn’t even ready (minimum viable product for me is a bicycle model lap sim with track elevation, with trajectory optimisation).

Absolutely no GenAI is being used for this project (I still don’t have confidence in its output) - everything here is pure human-generated slop :)

Expect a lot of refactoring because I have no clue what I’m doing :D

# Overview

…this will get filled eventually (i hope)

# Modules

*Note: The links below don’t work due to the Notion Markdown export, see the documentation website linked above or the Markdown files in the Documentation folder. However, the docstrings in the code will always be the most up-to-date.*

[1 Utils](1%20Utils%20252e66e2bd3a80fb9319d451e51e4b1c.md)

[2 Track](2%20Track%202d4e66e2bd3a8026ae52c079d9cee20a.md)

[3 Trajectory OLD](3%20Trajectory%20OLD%20254e66e2bd3a8036a34cf73c2b42096f.md)

[3 Trajectory](3%20Trajectory%202d4e66e2bd3a80c5afb5eccf798ee1c2.md)

[4 Quasistatic Lap Sim](4%20Quasistatic%20Lap%20Sim%20251e66e2bd3a80da94d2e49af5e89506.md)

[5 Dynamic Post-Processor](5%20Dynamic%20Post-Processor%20251e66e2bd3a802f878cd7d9e741f36e.md)

[6 Optimisation](6%20Optimisation%20254e66e2bd3a80d492daf86206e8709c.md)

# Miscellaneous

## To Do

- Write a script that takes MoTeC csv exported AC telemetry and extract coordinate arrays of the track at coordinates of CG-projected and front tyre contact patches
- Rework the trajectory module
    - Calculate track limit violations in terms of area violated (i.e. area of the polygon enclosed by the violating part of the trajectory and the track limits

## Required External Libraries

**External Libraries**

- **SciPy** - https://docs.scipy.org/doc/scipy/
- **Shapely** - https://shapely.readthedocs.io/en/stable/
- **NumPy** - https://numpy.org/doc/stable/index.html
- **Matplotlib** - https://matplotlib.org/stable/api/pyplot_summary.html

## **Useful Links and Potential Additional Libraries**

- **Spline-based trajectory optimisation research paper -** https://arxiv.org/abs/2309.09186
- **Matplotlib live plotting in a separate process** - https://matplotlib.org/stable/gallery/misc/multiprocess_sgskip.html
- **Optimisation using genetic algorithm, particle swarm etc. -** https://pymoo.org/index.html
- **Python timing code snippet compute time** - https://docs.python.org/3/library/timeit.html
- **Python multiprocessing** - https://docs.python.org/3/library/multiprocessing.html
- **Python logging** - https://docs.python.org/3/library/logging.html
- **Python profiling** - https://docs.python.org/3/library/profile.html

## Conventions

### Style

**Line Length**

- For docstrings, 80 characters or less
- For code and comments, free to use as much as necessary - though recommended limit of 150 so that 1 line fits on my 14-inch laptop

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
    Short 1 or 2 sentence summary of the module/function, with a full stop.
    
    After a line break, write a more detailed description of the module/function if
    necessary. This can include example use cases. Full sentences should end in a
    full stop. Bullet points can be used. For functions, the section below must be
    included (unless the function has no arguments, no returned values, and no
    raised errors/exceptions).
    
    Args (IF THE FUNCTION TAKES ARGUMENTS):
        Function argument 1: Description of the argument. This should contain the
            data type(s) expected if the type hints are unclear (particularly for
            lists and arrays). If this runs over a new line, use a tab indent for
            subsequent lines of the description.
        Function argument 2: Don't forget the full stop.
    
    Returns (IF THE FUNCTION RETURNS SOMETHING):
        Describe the return of the function, including the data types(s) expected.
        If there are multiple returns, this should state "Returns a tuple of (a, b,
        ...)" and list the values returned similar to the Args section, such as
        below.
        
        a: Description of the returned value. Should contain the data type(s)
            expected.
        
        b: Similar things here.
        
        This section can also contain logic if the function return changes based on
        the arguments. Note that blank lines are necessary for the documentation
        tooltip to render line breaks properly.
    
    Raises (IF THE FUNCTION RAISES ANYTHING):
        Error1: Text printed out if the error is raised
        Error2: Same thing, if there are multiple possible errors that can be raised
    """
    ```
    

### Multiprocessing

To run a single lap sim including dynamic post-processor will be single-threaded only

Optimisation will utilise multiprocessing to run multiple lap sims concurrently

### Variable Conventions

All internal variables and calculations use SI units - unless otherwise stated, assume SI units

Dictionary keys use PascalCase unless units are applicable, in which case it follows the convention below

Variable name conventions:

- Units not applicable → camelCase
- Units applicable → 1st part describes the unit (lower and/or upper case), 2nd part is the quantity (in PascalCase)
- Multiple units applicable → 1st part describes the units with each unit separated by an underscore ‘_’, 2nd part is the quantity (in PascalCase)

**A**

- Angle (rad) - positive anti-clockwise unless heading angle, in which case clockwise
- Area (m^2)

**a**

- Acceleration (m/s^2) - positive forwards

**B**

- Boolean - 1 for true, 0 for false

**b**

- N/A

**C**

- Coefficient (1)

**c**

- N/A

**D**

- Density (kg/m^3)

**d**

- Time-derivative modifier (/s) - positive increasing with time

**E**

- Energy (J)

**e**

- Efficiency (1, as a ratio)

**F**

- Force (N)

**f**

- Frequency (Hz)

**G**

- N/A

**g**

- N/A

**H**

- N/A

**h**

- Relative height (m) - positive higher

**I**

- Current (A)

**i**

- N/A

**J**

- N/A

**j**

- N/A

**K**

- Spring constant (N/m)

**k**

- Curvature (1/m) - positive curving to the left

**L**

- N/A

**l**

- Length (m)

**M**

- Moment/torque (Nm) - positive for forwards driving torque, negative for braking torque, otherwise follows right-hand rule

**m**

- Mass (kg)

**N**

- N/A

**n**

- Number (1)

**O**

- N/A

**o**

- N/A

**P**

- Power (W)

**p**

- Pressure (Pa)

**Q**

- N/A

**q**

- N/A

**R**

- Resistance (ohm)

**r**

- Ratio (1)

**S**

- N/A

**s**

- Distance/displacement (m) - positive in the direction of forwards travel

**T**

- Temperature (K)

**t**

- Time (s) - positive forwards in time

**U**

- N/A

**u**

- N/A

**V**

- Volume (m^3)
- Voltage (V)

**v**

- Speed (m/s)

**W**

- N/A

**w**

- Angular velocity (rad/s) - positive anti-clockwise

**X**

- N/A

**x**

- x coordinate (m) - positive forwards if car coordinates, positive east if track coordinates

**Y**

- N/A

**y**

- y coordinate (m) - positive left if car coordinates, positive north if track coordinates

**Z**

- N/A

**z**

- z coordinate (m) - positive upwards
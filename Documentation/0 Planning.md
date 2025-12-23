# 0 Planning

# Modules

[1 Utils](1%20Utils%20252e66e2bd3a80fb9319d451e51e4b1c.md)

[2 Track](2%20Track%20251e66e2bd3a80c4b32ecc2949b3cafa.md)

[3 Trajectory](3%20Trajectory%20254e66e2bd3a8036a34cf73c2b42096f.md)

[4 Quasistatic Lap Sim](4%20Quasistatic%20Lap%20Sim%20251e66e2bd3a80da94d2e49af5e89506.md)

[5 Dynamic Post-Processor](5%20Dynamic%20Post-Processor%20251e66e2bd3a802f878cd7d9e741f36e.md)

[6 Optimisation](6%20Optimisation%20254e66e2bd3a80d492daf86206e8709c.md)

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
- **If I try using Scipy Optimize again -** https://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function
- **If I wanna optimise using genetic algorithm/particle swarm/etc. (trajectory and/or setup) -** https://pymoo.org/index.html
- **Spline-based trajectory optimisation research paper -** https://arxiv.org/pdf/2309.09186

## Conventions

**Multiprocessing**

To run a single lap sim including dynamic post-processor will be single-threaded

Optimisation can then utilise multiprocessing to run multiple lap sims concurrently

**Variable Conventions**

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
| e |  |
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
| n |  |
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
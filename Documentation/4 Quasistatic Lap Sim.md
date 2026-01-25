# 4 Quasistatic Lap Sim

Plan is develop this in the following steps:

- Point mass implemented in Python, in 3D with a basic “tyre” model to capture induced rolling resistance from slip angle - mostly for me to have some maths practice and understand the basic equations
- Point mass implemented in Modelica - replicating the Python implementation to validate Modelica FMU for the vehicle model, which will make extensions to the vehicle model significantly easier
- Bicycle model implemented in Modelica
- 4-tyre model implemented in Modelica
- more complex stuff all in Modelica, ideally I can reuse the same vehicle model for both QS and dynamic modules, just with different wrappers

At the edge of the GGV envelope if:

- Max throttle input
- Max brake input
- Any tyre has a gradient ratio of 0
    - Note that similarly this can be gradient ratio at a defined limit, or stability metric at a defined limit

Therefore, at the edge of the GGV envelope when the minimum of:

- Throttle max threshold - throttle input
- Brake max threshold - brake input
- FL tyre gradient ratio (both long and lat)
- FR tyre gradient ratio (both long and lat)
- RL tyre gradient ratio (both long and lat)
- RR tyre gradient ratio (both long and lat)

equals 0 → so a root finding algorithm can be used which should be faster than a general minimisation algorithm (unless there’s a possibility for weird edge cases)

# PointMass

**Vehicle States Dictionary**

Contains:

- All vehicle states from the vehicle model

**Results Dictionary**

Contains:

- All vehicle states from the vehicle model
- Post-processed channels
- Track properties required to run the dynamic post-processor (or maybe it would be better to omit this and just pass in the Trajectory object)

---

# BicycleModel

Aero model defined with aero maps and use linear interpolation (if out of bounds then use nearest neighbour interpolation)

Use dimensions:

- Wing angles (may or may not exist, may also be multiple)
- Front ride height
- Rear ride height
- Roll angle
- Yaw angle relative to oncoming air
- Relative air speed

And calculate:

- Front downforce (account for drag centre of pressure)
- Rear downforce (account for drag centre of pressure)
- Drag
- Side force (acting on CG)
- Aero yaw moment

QS ride model solve for 0 force residuals given the pitch acceleration of the chassis dictated by the trajectory vertical curvature and longitudinal acceleration
# 4 Quasistatic Lap Sim

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

equals 0 → so a root finding algorithm can be used which should be faster than a general minimisation algorithm

# PointMass

**Vehicle States Dictionary**

**Results Dictionary**

| *Key* | *Value* |
| --- | --- |
| Time |  |
| Distance |  |

+ all the keys in [**Vehicle States Dictionary**](https://www.notion.so/Vehicle-States-Dictionary-251e66e2bd3a8042a499d13ab916ed6a?pvs=21), where each key’s value is an array with each item corresponding to each time/distance value

also include track properties? z height, slope, front/rear track camber - so the lap sim result can be passed straight to the ride/aero sim

---

# BicycleModel

Aero models define with aero maps and use linear interpolation (if out of bounds then use nearest neighbour interpolation)

Use dimensions:

- Wing angles (may or may not exist, may also be multiple)
- Front ride height
- Rear ride height
- Roll angle
- Yaw angle
- Vehicle speed

And calculate:

- Front downforce (account for drag centre of pressure)
- Rear downforce (account for drag centre of pressure)
- Drag
- Side force (acting on CG)
- Aero yaw moment

QS ride model solve for 0 force residuals given the pitch acceleration of the chassis dictated by the trajectory dSlope/ds and longitudinal acceleration
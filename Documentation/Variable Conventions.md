# Dictionary Keys

PascalCase unless units are applicable, in which case it follows the [Unit Prefixes](#Unit%20Prefixes) conventions

---

# Units

All internal variables and calculations use SI units - unless otherwise stated, assume SI units

---

# Variable Name Conventions

Order of variable naming parts follows the hierarchy:
1. Unit, if applicable
2. Quantity
3. Direction (”Left”, “FL” etc.)
4. Modifiers (”Soft”, “Filt” etc.)

|    Variable type     | Variable name convention                                                                         |
| :------------------: | ------------------------------------------------------------------------------------------------ |
| Units not applicable | `camelCase`                                                                                      |
|   Unit applicable    | First part is the [unit prefix](#Unit%20Prefixes)<br>Second part is the quantity in `PascalCase` |

---

# Unit Prefixes

Coordinate vectors concatenate the relevant coordinate prefixes (e.g. `xy` or `xyz`)

| Prefix | Unit(s)                                   | Convention                                                                                                                                                                               |
| :----: | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `A`   | Angle ($rad$)<br><br><br><br>Area ($m^2$) | Follows right-hand rule<br>- Yaw: Positive anti-clockwise<br>- Pitch: Positive front-down<br>- Roll: Positive right-down<br>                                                             |
|  `a`   | Acceleration ($m/s^2$)                    | Positive forwards                                                                                                                                                                        |
|  `B`   | Boolean (-)                               | 1 for `True`, 0 for `False`                                                                                                                                                              |
|  `b`   | -                                         |                                                                                                                                                                                          |
|  `C`   | Coefficient ($1$)                         |                                                                                                                                                                                          |
|  `c`   | -                                         |                                                                                                                                                                                          |
|  `D`   | Density ($kg/m^3$)                        |                                                                                                                                                                                          |
|  `d`   | Derivative modifier (-)                   | Time derivative ($d\over dt$): Append the prefix for the unit of the quantity<br><br>==think of how to distinguish between d/dt and d/ds<br>and other derivatives wrt other quantities   |
|  `E`   | Energy ($J$)                              |                                                                                                                                                                                          |
|  `e`   | Efficiency ($1$)                          | Given as a ratio                                                                                                                                                                         |
|  `F`   | Force ($N$)                               |                                                                                                                                                                                          |
|  `f`   | Frequency ($Hz$)                          |                                                                                                                                                                                          |
|  `G`   | -                                         |                                                                                                                                                                                          |
|  `g`   | -                                         |                                                                                                                                                                                          |
|  `H`   | -                                         |                                                                                                                                                                                          |
|  `h`   | Relative height ($m$)                     | Positive above                                                                                                                                                                           |
|  `I`   | Current ($A$)                             |                                                                                                                                                                                          |
|  `i`   | -                                         |                                                                                                                                                                                          |
|  `J`   | -                                         |                                                                                                                                                                                          |
|  `j`   | -                                         |                                                                                                                                                                                          |
|  `K`   | Spring constant ($N/m$)                   |                                                                                                                                                                                          |
|  `k`   | Curvature ($1/m$)                         | Follows right-hand rule<br>- Lateral: Positive left<br>- Vertical: Positive down                                                                                                         |
|  `L`   | -                                         |                                                                                                                                                                                          |
|  `l`   | Length ($m$)                              |                                                                                                                                                                                          |
|  `M`   | Moment/torque ($Nm$)                      | Positive forwards-driving (negative braking)                                                                                                                                             |
|  `m`   | Mass ($kg$)                               |                                                                                                                                                                                          |
|  `N`   | Number ($1$)                              |                                                                                                                                                                                          |
|  `n`   | -                                         |                                                                                                                                                                                          |
|  `O`   | -                                         |                                                                                                                                                                                          |
|  `o`   | -                                         |                                                                                                                                                                                          |
|  `P`   | Power ($W$)                               |                                                                                                                                                                                          |
|  `p`   | Pressure ($Pa$)                           |                                                                                                                                                                                          |
|  `Q`   | -                                         |                                                                                                                                                                                          |
|  `q`   | -                                         |                                                                                                                                                                                          |
|  `R`   | Resistance ($\Omega$)                     |                                                                                                                                                                                          |
|  `r`   | Ratio (1)                                 |                                                                                                                                                                                          |
|  `S`   |                                           |                                                                                                                                                                                          |
|  `s`   | Distance/displacement ($m$)               | Positive forwards ==is this also for suspension travel?==                                                                                                                                |
|  `T`   | Temperature ($K$)                         |                                                                                                                                                                                          |
|  `t`   | Time ($s$)                                |                                                                                                                                                                                          |
|  `U`   | -                                         |                                                                                                                                                                                          |
|  `u`   | -                                         |                                                                                                                                                                                          |
|  `V`   | Volume ($m^3$)<br>Voltage ($V$)           |                                                                                                                                                                                          |
|  `v`   | Speed ($m/s$)                             | Generic speed: Prefix `v`<br>Speed in a coordinate-aligned direction: Append the coordinate prefix (e.g. `vx`)<br>Velocity vector: Append the coordinate prefixes (e.g. `vxy` or `vxyz`) |
|  `W`   | -                                         |                                                                                                                                                                                          |
|  `w`   | Angular velocity ($rad/s$)                | Follows right-hand rule<br>- Yaw: Positive anti-clockwise<br>- Pitch: Positive front-down<br>- Roll: Positive right-down<br>- Wheels: Positive forwards-rolling                          |
|  `X`   | -                                         |                                                                                                                                                                                          |
|  `x`   | x coordinate ($m$)                        | Follows right-hand rule<br>- Vehicle coordinates: Positive forwards<br>- Track coordinates: Positive east                                                                                |
|  `Y`   | -                                         |                                                                                                                                                                                          |
|  `y`   | x coordinate ($m$)                        | Follows right-hand rule<br>- Vehicle coordinates: Positive left<br>- Track coordinates: Positive north                                                                                   |
|  `Z`   | -                                         |                                                                                                                                                                                          |
|  `z`   | x coordinate ($m$)                        | Follows right-hand rule<br>- Vehicle coordinates: Positive up<br>- Track coordinates: Positive up                                                                                        |

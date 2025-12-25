# 5 Dynamic Post-Processor

Feeds the quasistatic lap sim results into a dynamic model to estimate dynamic effects like:

- Ride
- Aerodynamic forces
- Thermals

As part of the post-processing, calculates the modifiers at each point in the trajectory required in the quasistatic lap sim to achieve similar behaviour for:

- Tyre grip
- Aerodynamic forces

Proof-of-concept just do a point-mass model with sprung and unsprung masses, and suspension and tyre spring and damper

Create the model in Modelica, then export as an FMU to interact through Python - use Runge-Kutta 4th order, timestep somewhere from 0.0005 to 0.001 seconds (Red Bull uses 0.0005 otherwise it’s numerically unstable)
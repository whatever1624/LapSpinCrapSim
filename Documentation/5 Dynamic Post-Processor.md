# 5 Dynamic Post-Processor

Feeds the quasistatic lap sim results into a dynamic model to estimate dynamic effects like:

- Ride
- Aero
- Thermals (maybe, idk if it’s important for me)

As part of the post-processing, calculates the (low-pass filtered/gaussian smoothed) modifiers at each point in the trajectory required in the quasistatic lap sim to achieve similar behaviour for:

- Tyre grip
- Aerodynamic forces
- whatever else there is

These modifiers can then be fed back into another iteration of the quasistatic lap sim to more closely approximate the dynamic behaviour of the car

Repeat a few times/until they converge

Proof-of-concept can be a lumped model with sprung and unsprung masses, and suspension and tyre springs and dampers

Create the model in Modelica, then export as an FMU to interact through Python - use Runge-Kutta 4th order, timestep somewhere from 0.0005 to 0.001 seconds (ride models I’m interacting with at work use 0.0005 otherwise it’s numerically unstable)
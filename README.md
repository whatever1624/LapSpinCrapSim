# Documentation Homepage

*Note: This documentation is created and maintained in Obsidian - syntax differences with GitHub-flavoured Markdown may cause rendering differences in the GitHub preview*

**GitHub Repo:** https://github.com/whatever1624/LapSpinCrapSim

> [!tip] First GitHub repo yippee!! 
> This is a solo project which I started at the start of the 2025 F1 summer shutdown as a passion project (also to learn how to use Git but that got procrastinated to weeks after shutdown), and I’ve been working on it sporadically since then.
> 
> The goal of this project is to be able to optimise trajectory, energy management, setup, and other parameters, using an optimal control problem formulation
> 
> Depending on the computation speed, the optimal control problem *may* get extended to include higher-order dynamics, and track representation *may* get extended to model high-fidelity kerbs - otherwise the optimisation of suspension setup options would require a ride sim in the loop
> 
> Currently the minimum viable product for me is a bicycle-model lap sim including track elevation and trajectory optimisation - but hopefully this project will extend to a 4-wheel vehicle model with suspension and wheelspin dynamics
> 
> Absolutely no GenAI is being used (one of my project goals is to develop my understanding, and also I still don’t fully trust its output) - everything here is pure human-generated slop so expect a lot of refactoring because I have no clue what I’m doing :D

---

# Overview

*...this will get filled eventually (i hope)*

The lap sim formulation as an optimal control problem is heavily referenced from the PhD work of an ex-colleague during my industrial placement year at Red Bull Racing: *Optimal Control of Vehicle Systems* (Giacomo Perantoni, 2013)

---

# Modules

- 1 Utils
	- [1.1 Helper Functions](1.1%20Helper%20Functions.md)
	- [1.2 Spline Utils](1.2%20Spline%20Utils.md)
	- 1.3 FMU Interface
- 2 Track
	- [2.1 Track](2.1%20Track.md)
	- [2.2 Surface](2.2%20Surface.md)
	- [2.3 Trajectory](2.3%20Trajectory.md)
- 3 Component Models
	- 3.1 Aero
	- 3.2 Engine
	- [3.3 Suspension](3.3%20Suspension.md)
	- [3.4 Tyre](3.4%20Tyre.md)
- 4 Vehicle Models
	- [4.1 Point Mass](4.1%20Point%20Mass.md)
	- 4.2 Bicycle
	- 4.3 4-Wheel ==(rigid body without suspension/tyre compliances, rephrase it better)==
	- 4.4 4-Wheel with Suspension
- 5 Point Mass Lap Sim ==(Dev Intermediate Step)==
	- 5.1 Python 2-Phase Solver
	- 5.2 FMU 2-Phase Solver
	- 5.3 Optimal Control Problem
- 6 Dynamic Lap Sim ==(is the development after 5.3 to go full dynamic lap sim then simplify that back down to quasistatic?)==
- 7 Quasistatic Lap Sim ==(or extend to a quasistatic bicycle model then again to a dynamic lap sim)==

==is the development after 5.3 to extend to a quasistatic bicycle model? or go full dynamic lap sim, then simplify that back down to quasistatic?==

==is it possible to selectively remove certain dynamic constraints to reduce the problem to quasistatic (e.g. remove suspension dynamics but still keep its quasistatic impacts, while still keeping speed/path dynamics) - i think this is just setting the relevant rows of the dynamics equality constraint vector to $0=\boldsymbol\psi_i$ instead of the continuity $0=\overline{\boldsymbol x}_i-\overline{\boldsymbol x}_{i-1}-\boldsymbol\psi_i(\overline{\boldsymbol x}_{i-1},B_i,\boldsymbol p)$ ?==

---

# Required External Libraries

- **NumPy** - https://numpy.org/doc/stable/index.html
- **SciPy** - https://docs.scipy.org/doc/scipy/
- **Matplotlib** - https://matplotlib.org/stable/api/pyplot_summary.html
- **PyFMI** *(to be implemented)* - https://github.com/modelon-community/PyFMI
- **cyipopt** *(to be implemented)* - https://cyipopt.readthedocs.io/stable/index.html

---

# Conventions

- [Style Conventions](Style%20Conventions.md)
- [Variable Conventions](Variable%20Conventions.md)

---

# To Do

- Massive refactor of LITERALLY EVERYTHING, transferring from previous Notion documentation
- Have the option of generating a trajectory from xyz coordinates (i.e. from telemetry) - will have to be heavily low-pass filtered if used for QS
- Write up [Sim Type Wishlist](Sim%20Type%20Wishlist.md) and transfer to main documentation
- When I get to it, the FMU export (model exchange) must export with analytical jacobian
	- Likely worth making “Reduced” (bare minimum outputs) versions of any models requiring the analytical jacobian so that the problem can be solved with the “Reduced” model which inherently should have a much smaller jacobian, then post-processed afterwards by passing the same points into the full model to get the full output
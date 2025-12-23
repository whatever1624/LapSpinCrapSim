# 6 Optimisation

---

# Trajectory Optimisation

*Note: Maybe I just need to find better parameters but SciPy optimisation doesn’t work that well - tried all the ones that allow an initial guess argument (minimise, basinhopping, dual_annealing)*

Try SciPy optimisation again but this time:

- Actually have a good continuous track limits penalty function
- Try all the different minimise() methods available
- Try setting tight bounds for every control point (e.g. plus minus 10m for every control point parameter), then doing minimise() for 1-2 iterations, then shift the bounds again (still plus minus 10m but for the new slightly more optimal control points) and repeat

If SciPy optimisation doesn’t work again then try writing a custom optimisation function based on gradient descent

- Starts with an initial perturb step size
- While optimising:
    - For each trajectory spline control point, perturbs up/down/left/right by the perturb step size
        - Evaluates objective function for each perturbed control point (some combination of the objective + punishment for track limit violations)
        - Updates the control point coordinates if a perturbed position performs better
    - If a pass over all control points didn’t change the coordinates of any control points, reduce the perturb step size
    - Repeat

### To Do

- Make an objective function for track centreline
    - Base “centreline” off the track limits gates
    - Find the centre point of each track limits gate (maybe calculate this when generating the trajectory and add it to the trajectory dictionary)
    - Make a LineString from the trajectory then iterate over all the track limits gates and use the shapely distance function
    - Objective function probably like sum of (distance error ^ exp) - experiment with different exponents

### Ideas

- Try perturbing based on the trajectory from the start of the loop iteration (i.e. only update control points after the loop) - this will probably cause bad track limits violations tho
- Try randomising the order that control points are optimised with each pass
- Try randomising the angle that the “perturb cross” is at (i.e. rotate the cross)
- Try using a different number of perturb directions (e.g. 8-pointed cross)
- Try having a small chance that a sub-optimal perturbation is accepted, or that a better perturbation isn’t accepted
- Try doing a little perturb of all coordinates in a random direction every so often

---

# Setup Optimisation

This will need a lot of collaboration between the quasistatic lap sim and the dynamic post-processor to find the tradeoffs particularly for ride

---

# Energy Management Optimisation

Includes electric motor deployment, full-throttle electric motor harvesting, and lift-and-coast

---

# Integrated Optimisation

Optimise everything above all at once - probably will need to split it up so it optimises 1 “category” at a time, and keeps iterating over all categories
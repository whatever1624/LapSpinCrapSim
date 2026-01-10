"""
Main place to configure and run the lap sim.

However, currently just being used as a sandbox to test the lap sim modules.
"""
# Python standard libraries
...

# External libraries
import matplotlib.pyplot as plt

# Project python modules
from track import Track

ovalTestTrack = Track(r"C:\Users\Willow\Documents\Repos\LapSpinCrapSim\Tracks\OvalTestTrack")

for i, gate in enumerate(ovalTestTrack.gates):
    z = ovalTestTrack.getTrackZ(gate.xyMidpoint, i)
    n = ovalTestTrack.getTrackNormal(gate.xyMidpoint, i)
    gateStr = f"Gate {i} [{gate.xyMidpoint[0]:.1f}, {gate.xyMidpoint[1]:.1f}]"
    print(f"{gateStr}{" " * (30 - len(gateStr))}{z:.3f}\t{n}")
    plt.scatter(i, z)
plt.show()

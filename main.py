"""
Main place to configure and run the lap sim.

However, currently just being used as a sandbox to test the lap sim modules.
"""
# Python standard libraries
...

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from track import Track

ovalTestTrack = Track(r"C:\Users\Willow\Documents\Repos\LapSpinCrapSim\Tracks\OvalTestTrack", True, BDebug=False)

fig, axs = plt.subplots(2, 1, layout='constrained')
for i, gate in enumerate(ovalTestTrack.gates):
    z = ovalTestTrack.getTrackZ(gate.xyMidpoint, i)
    n = ovalTestTrack.getTrackNormal(gate.xyMidpoint, i)
    gateStr = f"Gate {i} [{gate.xyMidpoint[0]:.1f}, {gate.xyMidpoint[1]:.1f}]"
    print(f"{gateStr}{" " * (30 - len(gateStr))}{z:.3f}\t{n}")

    axs[0].plot(i, z, marker='.', ls='-')
    axs[1].plot(i, np.acos(n[2]) * 180 / np.pi, marker='.', ls='-')

axs[0].set_ylabel("Track z coordinate (m)")
axs[1].set_ylabel("Track normal angle from vertical (deg)")
axs[1].set_ylim([0, 90])
axs[1].set_xlabel("Gate index")
plt.show()

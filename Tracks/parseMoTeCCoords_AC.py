"""
Script to parse Assetto Corsa MoTeC telemetry csv export for coordinates.

This outputs the coordinate array files in csv format:
    -   'xyzCPFL.csv': Front left contact patch coordinates.
    -   'xyzCPFR.csv': Front right contact patch coordinates.
    -   'xyzGroundF.csv': Ground-projected front ride height measurement point coordinates.

The option 'shiftCoords' will make all 3 coordinate arrays start on the same
line passing through the CG - either by rolling the contact patch coordinates
(if closed track), or discarding the first and last parts of the relevant
coordinates (if not closed track).
"""
# Python standard libraries
import os

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from Utils import utils
from track import CLOSED_TRACK_THRESHOLD_DISTANCE

# Telemetry file and coordinates save settings
telemFileName = r"r8 suzuka.csv"
saveFolder = r""
BRollCoords = True      # Whether to roll coordinates so that all 3 coordinate arrays start/finish roughly where the CG started/finished
                        # Note: Only rolls coordinates if the track is closed (calculated from CLOSED_TRACK_THRESHOLD_DISTANCE of the CG coordinates)

# Vehicle configuration settings - Assetto Corsa vehicle data and setup
rCG = 0.46              # CG location, CG_LOCATION from suspensions.ini (ratio)
lWheelbase = 2.7        # Wheelbase, WHEELBASE from suspensions.ini (m)
lTrackF = 1.680         # Front track width, TRACK from suspensions.ini (to consider the full width of the car, include tyre width to this) (m)
hOffsetF = -0.110       # Offset from CG for suspension component heights, BASEY from suspensions.ini (m)
hPickupF = -0.350       # Offset from CG for front ride height, PICKUP_FRONT_HEIGHT from car.ini or 0 if not present (m)
lRodSetupF = 7          # Front rod length, ROD_LENGTH_XX from the setup (mm - this is converted to m automatically in the script)

# Parse csv telemetry file
print("\nParsing telemetry file")
replaceQuotes = lambda x: x.replace('\"', '')
channels = np.loadtxt(telemFileName, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=13, max_rows=1)
units = np.loadtxt(telemFileName, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=14, max_rows=1)
data = np.loadtxt(telemFileName, dtype=float, delimiter=',', converters=replaceQuotes, skiprows=17, max_rows=1000)

# Convert data to SI units and organise into a dictionary by channel name
print("Processing telemetry")
telemDict = {}
for i, channel in enumerate(channels):
    telemDict[channel] = utils.convertUnits(data[:, i], units[i])

# Get the channels and data required
xCG = telemDict['Car Coord X']
yCG = telemDict['Car Coord Y']
zCG = telemDict['Car Coord Z']
aPitch = telemDict['Chassis Pitch Angle']
aRoll = telemDict['Chassis Roll Angle'] * -1                            # AC has inverted roll convention
vxCG = telemDict['Chassis Velocity X']
vyCG = telemDict['Chassis Velocity Y']
hF = (telemDict['Ride Height FL'] + telemDict['Ride Height FR']) / 2    # Technically this is unecessary since FL and FR both output the same data
lSuspTravelFL = telemDict['Suspension Travel FL']
lSuspTravelFR = telemDict['Suspension Travel FR']
lTyreLoadedRadiusFL = telemDict['Tire Loaded Radius FL']
lTyreLoadedRadiusFR = telemDict['Tire Loaded Radius FR']

# Get derived data and channels
nData = len(xCG)
zeros = np.zeros_like(nData)
xyzCG = np.vstack((xCG, yCG, zCG)).T
lAxleF = lWheelbase * (1 - rCG)     # Distance from CG to front axle
lRodSetupF *= 1e-3                  # Convert to SI units (mm to m)

# Calculate the chassis yaw angle from the direction of travel in world coordinates and in car coordinates
print("Calculating coordinates")
aSlipChassis = np.atan2(vyCG, vxCG)
avxyCG = np.atan2(np.gradient(yCG), np.gradient(xCG))
aYaw = avxyCG + aSlipChassis

# Calculate the ground-projected front ride height coordinates
# Get vector (car reference) from CG to front ride height measurement point
xyzVecGroundF = np.full_like(xyzCG, (lAxleF, 0, hPickupF))
xyzVecGroundF[:, 2] -= hF
# Rotate to track reference
xyzVecGroundF = np.array([utils.rotateVector3D(xyzVecGroundF[i], aYaw[i], aPitch[i], aRoll[i]) for i in range(nData)])
# Add to CG coordinate to get front left contact patch coordinates in track reference frame
xyzGroundF = xyzCG + xyzVecGroundF

# Calculate the front left contact patch coordinate
# Get vector (car reference) from CG to front left contact patch
xyzVecCGCPFL = np.full_like(xyzCG, (lAxleF, lTrackF / 2, hOffsetF + lRodSetupF))
xyzVecCGCPFL[:, 2] += lSuspTravelFL - lTyreLoadedRadiusFL
# Rotate to track reference
xyzVecCGCPFL = np.array([utils.rotateVector3D(xyzVecCGCPFL[i], aYaw[i], aPitch[i], aRoll[i]) for i in range(nData)])
# Add to CG coordinate to get front left contact patch coordinates in track reference frame
xyzCPFL = xyzCG + xyzVecCGCPFL

# Calculate the front right contact patch coordinate
# Get vector (car reference) from CG to front right contact patch
xyzVecCGCPFR = np.full_like(xyzCG, (lAxleF, -lTrackF / 2, hOffsetF + lRodSetupF))
xyzVecCGCPFR[:, 2] += lSuspTravelFR - lTyreLoadedRadiusFR
# Rotate to track reference
xyzVecCGCPFR = np.array([utils.rotateVector3D(xyzVecCGCPFR[i], aYaw[i], aPitch[i], aRoll[i]) for i in range(nData)])
# Add to CG coordinate to get front right contact patch coordinates in track reference frame
xyzCPFR = xyzCG + xyzVecCGCPFR

# Roll coordinates backwards by distance to the front axle, only if the setting is enabled and the track is calculated to be closed
if BRollCoords:
    if np.linalg.norm(xyzCG[-1] - xyzCG[0]) <= CLOSED_TRACK_THRESHOLD_DISTANCE:
        print("Rolling coordinates")
        sCPFL = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzCPFL, axis=0), axis=1)))
        sCPFR = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzCPFR, axis=0), axis=1)))
        sFGround = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzGroundF, axis=0), axis=1)))
        xyzCPFL = np.roll(xyzCPFL, -np.argmin(np.abs(sCPFL - lAxleF)), axis=0)
        xyzCPFR = np.roll(xyzCPFR, -np.argmin(np.abs(sCPFR - lAxleF)), axis=0)
        xyzGroundF = np.roll(xyzGroundF, -np.argmin(np.abs(sFGround - lAxleF)), axis=0)
    else:
        print("BRollCoords setting enabled, but track is not closed")

# Save coordinates to csv files
print("Saving coordinates to csv files")
np.savetxt(os.path.join(saveFolder, "xyzCPFL.csv"), xyzCPFL, fmt='%1.18f', delimiter=',')
np.savetxt(os.path.join(saveFolder, "xyzCPFR.csv"), xyzCPFR, fmt='%1.18f', delimiter=',')
np.savetxt(os.path.join(saveFolder, "xyzGroundF.csv"), xyzGroundF, fmt='%1.18f', delimiter=',')

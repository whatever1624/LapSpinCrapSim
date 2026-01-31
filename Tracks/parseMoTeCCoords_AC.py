"""
Script to parse Assetto Corsa MoTeC telemetry csv export for coordinates.

This outputs the coordinate array files in csv format:
    ...TODO

The option 'shiftCoords' will make all 3 coordinate arrays start on the same
line passing through the CG - either by rolling the contact patch coordinates
(if closed track), or discarding the first and last parts of the relevant
coordinates (if not closed track).

TODO: Add functionality to automatically generate gates for start/finish, DRS,
    sector timing, speed limit (pits) etc.
"""
# Python standard libraries
import os

# External libraries
import numpy as np

# Project python modules
from Utils.typeAliases import NDArrayFloat1D
from Utils import utils
from track import CLOSED_TRACK_THRESHOLD_DISTANCE

# Folder path settings
telemFolder = r"C:\Users\Willow\Documents\Repos\LapSpinCrapSim\Tracks\BrandsHatchIndy\Telemetry"
saveFolder = r"C:\Users\Willow\Documents\Repos\LapSpinCrapSim\Tracks\BrandsHatchIndy"

# Coordinate output settings
BRollCoords = True      # Whether to roll coordinates so that all 3 coordinate arrays start/finish roughly where the CG started/finished
                        # Note: Only rolls coordinates if the track is closed (calculated from CLOSED_TRACK_THRESHOLD_DISTANCE of the CG coordinates)
BAdjustzGroundF = True  # Whether to adjust the z coordinate calculated of the ground-projected front ride height measurement point so that its
                        # average is equal to the average of the front left and front right contact patch z coordinates over the telemetry

# Vehicle configuration settings - Assetto Corsa vehicle data and setup
rCG = 0.41              # CG location, CG_LOCATION from suspensions.ini (ratio)
lWheelbase = 2.650      # Wheelbase, WHEELBASE from suspensions.ini (m)
lTrackF = 1.515 + 0.180 # Front track width, TRACK from suspensions.ini (add 1 front tyre width to consider the full width of the car) (m)
hOffsetF = -0.020       # Offset from CG for suspension component heights, BASEY from suspensions.ini (m)
hPickupF = -0.248       # Offset from CG for front ride height, PICKUP_FRONT_HEIGHT from car.ini or 0 if not present (m)
lRodSetupF = 40         # Front rod length, ROD_LENGTH_XX from the setup (mm - this is converted to m automatically in the line below)
lRodSetupF *= 1e-3


def getCoordsFromTelem(telemFile: str | os.PathLike[str]):
    # Parse csv telemetry file
    replaceQuotes = lambda x: x.replace('\"', '')
    channels = np.loadtxt(telemFile, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=13, max_rows=1)
    units = np.loadtxt(telemFile, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=14, max_rows=1)
    data = np.loadtxt(telemFile, dtype=float, delimiter=',', converters=replaceQuotes, skiprows=17)

    # Convert data to SI units and organise into a dictionary by channel name
    telemDict = {channel: utils.convertUnits(data[:, i], units[i]) for i, channel in enumerate(channels)}

    # Get the channels and data required
    xCG = telemDict['Car Coord X']
    yCG = telemDict['Car Coord Y']
    zCG = telemDict['Car Coord Z']
    aPitch = telemDict['Chassis Pitch Angle']
    aRoll = telemDict['Chassis Roll Angle']
    vxCG = telemDict['Chassis Velocity X']
    vyCG = telemDict['Chassis Velocity Y']
    hF = (telemDict['Ride Height FL'] + telemDict['Ride Height FR']) / 2    # Technically unnecessary since FL and FR both output the same data
    lSuspTravelFL = telemDict['Suspension Travel FL']
    lSuspTravelFR = telemDict['Suspension Travel FR']
    lTyreLoadedRadiusFL = telemDict['Tire Loaded Radius FL']
    lTyreLoadedRadiusFR = telemDict['Tire Loaded Radius FR']

    # Get derived data and channels
    nData = len(xCG)
    xyzCG = np.vstack((xCG, yCG, zCG)).T
    lAxleF = lWheelbase * (1 - rCG)     # Distance from CG to front axle
    aSlipChassis = np.atan2(vyCG, vxCG)
    avxyCG = np.atan2(np.gradient(yCG), np.gradient(xCG))
    aYaw = avxyCG + aSlipChassis

    def calcTrackCoord(xyzVecConstant: list[float] | NDArrayFloat1D,
                       zOffsetChannel: NDArrayFloat1D):
        """
        Calculates the ground-projected coordinate at a given location on the
        car, given its vector from the CG and channel of z height from the track
        at that point.

        # Calculate the ground-projected front ride height coordinates
        # Get vector (car reference) from CG to front ride height measurement point
        # Rotate to track reference
        # Add to CG coordinate to get front left contact patch coordinates in track reference frame

        TODO: Docstring
        """
        xyzVecs = np.full_like(xyzCG, xyzVecConstant)
        xyzVecs[:, 2] -= zOffsetChannel
        xyzVecs = np.array([utils.rotateVector3D(xyzVecs[i], aYaw[i], aPitch[i], aRoll[i]) for i in range(nData)])
        xyz = xyzCG + xyzVecs
        return xyz

    xyzGroundF = calcTrackCoord([lAxleF, 0, hPickupF], hF)
    xyzCPFL = calcTrackCoord([lAxleF, lTrackF / 2, hOffsetF + lRodSetupF], lTyreLoadedRadiusFL - lSuspTravelFL)
    xyzCPFR = calcTrackCoord([lAxleF, -lTrackF / 2, hOffsetF + lRodSetupF], lTyreLoadedRadiusFR - lSuspTravelFR)

    # Roll coordinates backwards by distance to the front axle, only if the setting is enabled and the track is calculated to be closed
    if BRollCoords:
        if np.linalg.norm(xyzCG[-1] - xyzCG[0]) <= CLOSED_TRACK_THRESHOLD_DISTANCE:
            sCPFL = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzCPFL, axis=0), axis=1)))
            sCPFR = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzCPFR, axis=0), axis=1)))
            sFGround = np.append(0, np.cumsum(np.linalg.norm(np.diff(xyzGroundF, axis=0), axis=1)))
            xyzCPFL = np.roll(xyzCPFL, -np.argmin(np.abs(sCPFL - lAxleF)), axis=0)
            xyzCPFR = np.roll(xyzCPFR, -np.argmin(np.abs(sCPFR - lAxleF)), axis=0)
            xyzGroundF = np.roll(xyzGroundF, -np.argmin(np.abs(sFGround - lAxleF)), axis=0)
        else:
            print("BRollCoords setting enabled, but track is not closed")

    # Adjust ground-projected front ride height measurement point coordinates so its average is equal to the average of the contact patch coordinates
    if BAdjustzGroundF:
        xyzGroundF[:, 2] += ((np.mean(xyzCPFL[:, 2]) + np.mean(xyzCPFR[:, 2])) / 2) - np.mean(xyzGroundF[:, 2])

    return xyzCPFL, xyzCPFR, xyzGroundF


# Get coordinates from telemetry
limitFileNames = ('LimitLeftSoft.csv', 'LimitRightSoft.csv', 'LimitLeftHard.csv', 'LimitRightHard.csv')
with os.scandir(telemFolder) as entries:
    for entry in entries:
        if entry.name in limitFileNames or (entry.name.startswith('Extra') and entry.name.endswith('.csv')):
            # Valid telemetry file, get coordinates as the tuple (xyzCPFL, xyzCPFR, xyzGroundF)
            xyzCoordsTuple = getCoordsFromTelem(entry)

            # Set csv file names
            base = 'xyzExtra' + entry.name[:-4] if entry.name.lower in limitFileNames else 'xyzExtra' + entry.name[5:-4]
            fileNames = [base + 'CPFL.csv', base + 'CPFR.csv', base + 'GroundF.csv']
            if entry.name in (limitFileNames[0], limitFileNames[2]):
                fileNames[0] = 'xyz' + entry.name
            elif entry.name in (limitFileNames[1], limitFileNames[3]):
                fileNames[1] = 'xyz' + entry.name

            # Save coordinates to csv files
            for i in range(len(xyzCoordsTuple)):
                np.savetxt(os.path.join(saveFolder, fileNames[i]), xyzCoordsTuple[i], fmt='%1.18f', delimiter=',')

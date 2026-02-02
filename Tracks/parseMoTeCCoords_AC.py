"""
Script to parse Assetto Corsa MoTeC telemetry csv export for coordinates.

Telemetry files that will be parsed into coordinates must be:
    -   A csv file (file extension .csv) in MoTeC export format.
    -   Named as one of the track limits 'LimitLeftSoft.csv',
        'LimitRightSoft.csv', 'LimitLeftHard.csv', 'LimitRightHard.csv'.
    -   Alternatively, named starting with the prefix 'Extra'.

The coordinate array csv files output will be in the format accepted by the
track module, and each telemetry file will generate 3 coordinate array files:
    -   Front left contact patch coordinate, filename ending with 'CPFL.csv'.
    -   Front right contact patch coordinate, filename ending with 'CPFR.csv'.
    -   Ground-projected front ride height pickup point coordinate, filename
        ending with 'GroundF.csv'.

The exceptions to the output naming are the track limit coordinate array csv
files, which are deduced as the front left/right contact patch coordinate given
the telemetry file name.

Note: It is recommended to save the MoTeC telemetry with the default/highest
resolution (most decimal places) - particularly for the track z coordinate
calculation.

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
lTrackF = 1.515         # Front track width, TRACK from suspensions.ini (add 1 front tyre width to consider the full width of the car) (m)
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
    APitch = telemDict['Chassis Pitch Angle'] * -1                          # AC has inverted pitch
    ARoll = telemDict['Chassis Roll Angle']
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
    lAxleF = lWheelbase * (1 - rCG)                             # Distance from CG to front axle
    AHeadingCG = np.atan2(np.gradient(-yCG), np.gradient(xCG))  # Heading angle of the CG velocity vector in track coordinates
    ASlipChassis = np.atan2(vyCG, vxCG)                         # Chassis slip angle in car coordinates

    def calcProjectedTrackCoords(xyzVecConstant: list[float] | NDArrayFloat1D,
                                 zOffsetChannel: NDArrayFloat1D) -> NDArrayFloat1D:
        """
        Calculates the ground-projected coordinate derived from a channel of
        z height from the track in car coordinates.

        First gets the vector in car reference from the CG to the measurement
        point. Then rotates the vector into the track reference. Finally adds
        the vector to the CG coordinate.

        Args:
            xyzVecConstant: Vector from the CG to the measurement point, in the
                car's reference frame and in the form [x, y, z]. This is a
                constant.
            zOffsetChannel: Channel of z height from the track, in the car's
                reference frame. This is positive if the car is higher off the
                track.

        Returns:
            2D NumPy array representing the ground-projected coordinates in
            track coordinates. Each coordinate is in the form [x, y, z].
        """
        xyzVecs = np.full_like(xyzCG, xyzVecConstant)
        xyzVecs[:, 2] -= zOffsetChannel
        for i in range(nData):
            xyzVecs[i] = utils.rotateVectorHeading(utils.rotateVector3D(xyzVecs[i], ARoll[i], APitch[i], ASlipChassis[i]), AHeadingCG[i])
        xyz = xyzCG + xyzVecs
        return xyz

    xyzGroundF = calcProjectedTrackCoords([lAxleF, 0, hPickupF], hF)
    xyzCPFL = calcProjectedTrackCoords([lAxleF, lTrackF / 2, hOffsetF + lRodSetupF], lTyreLoadedRadiusFL - lSuspTravelFL)
    xyzCPFR = calcProjectedTrackCoords([lAxleF, -lTrackF / 2, hOffsetF + lRodSetupF], lTyreLoadedRadiusFR - lSuspTravelFR)

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
            # Valid telemetry file
            print(f"Parsing telemetry file: {entry.name}")

            # Get the coordinate arrays as the tuple (xyzCPFL, xyzCPFR, xyzGroundF)
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

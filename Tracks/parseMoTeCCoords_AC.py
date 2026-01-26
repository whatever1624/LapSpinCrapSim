"""
Script to parse Assetto Corsa MoTeC telemetry csv export for coordinates.
"""
# Python standard libraries
import os

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Project python modules
from Utils.typeAliases import NDArrayFloat2D
from Utils import utils

# Telemetry file and save folder settings
telemFileName = r"r8 suzuka.csv"
saveFolder = r""

# Parse csv telemetry file
replaceQuotes = lambda x: x.replace('\"', '')
channels = np.loadtxt(telemFileName, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=13, max_rows=1)
units = np.loadtxt(telemFileName, dtype=str, delimiter=',', converters=replaceQuotes, skiprows=14, max_rows=1)
data = np.loadtxt(telemFileName, dtype=float, delimiter=',', converters=replaceQuotes, skiprows=17)

# Convert data to SI units and organise into a dictionary
telem = {}
for i, channel in enumerate(channels):
    telem[channel] = utils.convertUnits(data[:, i], units[i])

    channelUnit = f"{channel} ({units[i] if units[i] != '' else '-'})"
    print(f"{channelUnit}{' ' * (40 - len(channelUnit))}{data[0, i]}\t{telem[channel][0]}")

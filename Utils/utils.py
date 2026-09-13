"""
Collection of helper functions commonly used by the other modules
"""

# Python standard libraries
from typing import Literal

# External libraries
import scipy
import numpy as np

# Project python modules
from Utils.typealiases import Any, NDArrayFloat1D, NDArrayFloat2D, NDArrayNumber1D


def wrap(x: float | NDArrayFloat1D | NDArrayFloat2D,
         lowerBound: float,
         upperBound: float) -> float | NDArrayFloat1D:
    """
    Wraps value(s) between the lower (inclusive) and upper (exclusive) bounds

    Supports x as a NumPy array of values to be wrapped

    Args:
        x: Float or NumPy array of floats to wrap
        lowerBound: Lower bound (inclusive) for the wrapping
        upperBound: Upper bound (exclusive) for the wrapping

    Returns:
        If the argument x passed in was a float, returns the wrapped float of x

        If the argument x passed in was a NumPy array of floats, returns an array of floats in same shape where each
        element is wrapped.
    """
    return lowerBound + ((x - lowerBound) % (upperBound - lowerBound))


def linear_interp_extrap(x: float,
                         xp: list[float] | NDArrayNumber1D,
                         fp: list[float] | NDArrayNumber1D) -> float:
    """
    Linearly interpolate or extrapolate at the point x from the closest 2 data points defined by xp, fp

    Able to compute linear extrapolation unlike NumPy interp()
    For computing single points, faster than NumPy interp() if there is a reasonable chance (>15%) that the point will
    be calculated using the first or last pair of data points

    Args:
        x: x coordinate to evaluate the extrapolated function
        xp: x coordinates of the function. Must contain 2 or more elements and be monotonically increasing (though not
            validated explicitly)
        fp: Function values corresponding to the x coordinates of xp, must be the same size as xp

    Returns:
        Extrapolated function value at point x
    """
    # Find the 2 data points to use for the linear extrapolation
    if x <= xp[1]:
        xp = xp[:2]
        fp = fp[:2]
    elif x >= xp[-2]:
        xp = xp[-2:]
        fp = fp[-2:]
    else:
        # Point x is not within the first pair or last pair of data points, fallback to NumPy interp()
        return np.interp(x, xp, fp)

    # Calculate the linear extrapolation
    return fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])


def resample(signal: list[float] | NDArrayFloat1D,
             signalBase: list[float] | NDArrayFloat1D,
             fResample: float,
             BExtrapolate: bool = False) -> tuple[NDArrayFloat1D, NDArrayFloat1D]:
    """
    Resamples the signal at the frequency specified

    Args:
        signal: Signal, can be temporal or spatial
        signalBase: Base that the signal was sampled on (usually temporal or spatial), must be monotonically increasing
        fResample: Resample rate of the signal, in Hz (if temporal) or cycles/m (if spatial)
        BExtrapolate: Whether to (linearly) extrapolate the signal such that the resampled signal contains the full
            temporal or spatial range of the original signal
            If false, truncates the resampled signal at the closest resampling point to the end of the original signal

    Returns:
        Tuple of (signalResampled, resampledBase)

        signalResampled: Resampled signal

        resampledBase: New base of the resampled signal
    """
    # Create the new base of the resampled signal, only in the interpolating region
    dResample = 1 / fResample
    resampledBase = np.arange(signalBase[0], signalBase[-1], dResample)
    resampledBaseNext = resampledBase[-1] + dResample
    if resampledBaseNext < signalBase[-1]:
        resampledBase = np.append(resampledBase, resampledBaseNext)

    # Resample the signal
    signalResampled = np.interp(resampledBase, signalBase, signal)

    # Calculate the signal extrapolation
    if BExtrapolate:
        resampledBase = np.append(resampledBase, resampledBaseNext)
        signalResampled = np.append(signalResampled, linear_interp_extrap(resampledBaseNext, signalBase, signal))

    return signalResampled, resampledBase


def filt(signal: list[float] | NDArrayFloat1D,
         fSample: float,
         filtType: Literal['low', 'high', 'bandpass', 'bandstop'],
         fCutoff: float | list[float] | NDArrayFloat1D,
         nOrder: int) -> NDArrayFloat1D:
    """
    Filters the signal using a Butterworth filter

    Args:
        signal: Signal, can be temporal or spatial, but must be sampled at regular intervals
        fSample: Sample rate of the signal in Hz (if temporal) or cycles/m (if spatial)
        filtType: Type of filter, must be one of the strings supported by the btype argument for the SciPy butter()
            function - but to simplify, the options are 'low', 'high', 'bandpass', 'bandstop'
        fCutoff: Cutoff frequency (if low or high-pass filter) or frequencies in the form [fCutoffLow, fCutoffHigh]
            (if band-pass or band-stop filter)
            In Hz (if temporal) or cycles/m (if spatial)
        nOrder: Order of the filter

    Returns:
        Filtered signal, as a NumPy array
    """
    # Calculate the filter padlen as the number of samples in 5 cycles of the (lowest) cutoff frequency
    # This is to avoid flattened artefacts at the start/end of the signal with a low-pass filter
    fCutoffLower = fCutoff if np.isscalar(fCutoff) else min(fCutoff)
    padlen = int(min(np.ceil(5 * fSample / fCutoffLower), len(signal) - 1))

    # If there is a low-pass component to the filter, set the first and last points to the average of the first/last
    # half-cycle at the cutoff frequency
    if 'h' not in filtType:
        nPoints = min(int(fSample / fCutoffLower / 2), len(signal))
        signal[0] = np.mean(signal[:nPoints])
        signal[-1] = np.mean(signal[-nPoints:])

    # Filter the signal
    sos = scipy.signal.butter(nOrder, fCutoff, filtType, output='sos', fs=fSample)
    signalFilt = scipy.signal.sosfiltfilt(sos, signal, padlen=padlen)

    return signalFilt


def get_inds_without_consecutive_duplicates(arr: list[Any] | np.ndarray[tuple[Any, ...], np.dtype[Any]],
                                            axis: int = -1) -> np.ndarray[tuple[Any, ...], np.dtype[np.integer]]:
    """
    Returns the indexes of the array that omit elements that would cause consecutive duplicates

    Args:
        arr: List or array, where the data type must be compatible with the NumPy function diff()
        axis: Axis along which to check for consecutive duplicates, defaults to the last axis

    Returns:
        Array with the indexes that omit elements causing consecutive duplicates along the axis specified
    """
    diff = np.diff(arr, axis=axis).astype(np.bool)
    if diff.ndim > 1:
        diff = np.any(diff, axis=-1)
    return np.append(0, np.where(diff)[0] + 1)


def remove_consecutive_duplicates(arr: list[Any] | np.ndarray[tuple[Any, ...], np.dtype[Any]],
                                  axis: int = -1) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
    """
    Returns the array with consecutive duplicates removed along the axis specified

    Effectively a wrapper around getIndsWithoutConsecutiveDuplicates()

    Args:
        arr: List or array, where the data type must be compatible with the NumPy function diff()
        axis: Axis along which to check for consecutive duplicates, defaults to the last axis

    Returns:
        Array with the consecutive duplicates removed along the axis specified
    """
    return np.array(arr)[get_inds_without_consecutive_duplicates(arr, axis)]


def calc_side_of_line(xyPoint: list[float] | NDArrayFloat1D | tuple[float, float],
                      xyLineStart: NDArrayFloat1D,
                      xyLineEnd: NDArrayFloat1D) -> float:
    """
    Finds which side of the line that the specified point is

    Supports both [x, y] and [x, y, z] coordinates as inputs - but only computes in the 2D [x, y] plane

    Args:
        xyPoint: Coordinate of the point, where index 0 is x and index 1 is y
        xyLineStart: Coordinate of the start of the line, where index 0 is x and index 1 is y
        xyLineEnd: Coordinate of the end of the line, where index 0 is x and index 1 is y

    Returns:
        0 if all coordinates are collinear (on the same straight line)

        >0 if the point is on the right of the line

        < 0 if the point is on the left of the line
    """
    return (((xyPoint[0] - xyLineStart[0]) * (xyLineEnd[1] - xyLineStart[1]))
            - ((xyPoint[1] - xyLineStart[1]) * (xyLineEnd[0] - xyLineStart[0])))


def convert_units(data: float | NDArrayFloat1D,
                  currentUnit: str,
                  newUnit: str = '') -> float | NDArrayFloat1D:
    """
    Convert data between units

    Args:
        data: Data to convert between units
        currentUnit: Unit of the data passed in
        newUnit: Unit to convert the data to - uses the SI unit if not provided

    Returns:
        Data converted to the new unit
        This will be a float if the argument data passed in was a float, or a 1D NumPy array of floats if the argument
        data passed in was a 1D NumPy array of floats

    Raises:
        ValueError1: 'newUnit' argument provided as '{newUnit}' but 'currentUnit' argument is blank
        ValueError2: 'newUnit' argument of '{newUnit}' not supported for 'currentUnit' argument of '{currentUnit}'
        ValueError3: '{currentUnit}' is not a supported unit by the convertUnits() function
    """
    # Each value in this dictionary is itself a dictionary, where each key-value pair is the unit and conversion
    # The conversion is a tuple in the form (multiplier, offset), which applied as (data * multiplier) + offset,
    # converts from the unit to SI
    # The SI unit will always have the conversion (1, 0)
    # The nested dictionaries are ordered such that SI units/SI prefixes are first,
    # and then ordered by the conversion factor
    SI = (1, 0)
    conversionsDict: dict[str, dict[str, tuple[float, float]]] = {
        'Angle': {'rad': SI,
                  'deg': (np.pi / 180, 0)},

        'Area': {'m2': SI,
                 'm^2': SI},

        'Acceleration': {'m/s2': SI,
                         'G': (9.81, 0)},

        'Density': {'kg/m3': SI},

        'Energy': {'J': SI,
                   'kJ': (1e3, 0),
                   'MJ': (1e6, 0)},

        'Force': {'N': SI},

        'Frequency': {'Hz': SI},

        'Length': {'mm': (1e-3, 0),
                   'm': SI,
                   'km': (1e3, 0)},

        'Torque': {'Nm': SI,
                   'N.m': SI},

        'Mass': {'kg': SI},

        'Power': {'W': SI,
                  'kW': (1e3, 0),
                  'hp': (745.699872, 0),
                  'HP': (745.699872, 0)},

        'Pressure': {'Pa': SI,
                     'kPa': (1e3, 0),
                     'bar': (1e5, 0),
                     'psi': (6894.76, 0),
                     'PSI': (6894.76, 0)},

        'Ratio': {'%': (0.01, 0),
                  '1': SI},

        'Temperature': {'K': SI,
                        'C': (1, 273.15),
                        'degC': (1, 273.15)},

        'Time': {'s': SI},

        'Volume': {'m3': SI,
                   'l': (1e3, 0),
                   'L': (1e3, 0)},

        'Speed': {'m/s': SI,
                  'km/h': (1 / 3.6, 0),
                  'kph': (1 / 3.6, 0)},

        'Angular velocity': {'rad/s': SI,
                             'deg/s': (np.pi / 180, 0),
                             'rpm': (np.pi / 30, 0),
                             'RPM': (np.pi / 30, 0)}}

    # Return the data unchanged if both newUnit and currentUnit are blank
    if currentUnit == '':
        if newUnit == '':
            return data
        else:
            raise ValueError(f"'newUnit' argument provided as '{newUnit}' but 'currentUnit' argument is blank")

    # Find the quantity (conversion dictionary to use) by matching the currentUnit argument
    for conversionDict in conversionsDict.values():
        if currentUnit in conversionDict:
            # Conversion dictionary found, convert the data to SI
            multiplier, offset = conversionDict[currentUnit]
            data = (data * multiplier) + offset

            # Check if the default SI unit should be returned (if newUnit is '', which it defaults to if not passed)
            if newUnit == '':
                return data

            # Get the conversion to the new unit and convert the data to the new unit if the conversion was found,
            # otherwise raise an error
            multiplier, offset = conversionDict.get(newUnit, (None, None))
            if multiplier is not None and offset is not None:
                return (data - offset) / multiplier
            else:
                raise ValueError(f"'newUnit' argument of '{newUnit}' not supported for 'currentUnit' argument of " +
                                 f"'{currentUnit}'")

    # Match not found for the currentUnit argument
    raise ValueError(f"'{currentUnit}' is not a supported unit by the convertUnits() function")


def rotate_vector_3D(xyz: NDArrayFloat1D,
                     ARoll: float,
                     APitch: float,
                     AYaw: float) -> NDArrayFloat1D:
    """
    Rotates a 3D vector in the form [x, y, z] by the roll, pitch and yaw angles, in that order

    This uses intrinsic rotations with Tait-Bryan angles
    See: https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    Args:
        xyz: NumPy array in the form [x, y, z] representing the 3D vector
        ARoll: Roll angle in radians to rotate the vector, right-side down (follows the right-hand rule)
            This rotation is applied first
        APitch: Pitch angle in radians to rotate the vector, pitched down (follows the right-hand rule)
            This rotation is applied second
        AYaw: Yaw angle in radians to rotate the vector, anti-clockwise (follows the right-hand rule)
            This rotation is applied last

    Returns:
        NumPy array representing the rotated vector, in the form [x*, y*, z*]
    """
    cr = np.cos(ARoll)
    sr = np.sin(ARoll)
    cp = np.cos(APitch)
    sp = np.sin(APitch)
    cy = np.cos(AYaw)
    sy = np.sin(AYaw)
    """
    # Separated rotation matrices for debugging
    rotMatrixRoll = np.array([[1, 0, 0],
                              [0, cr, -sr],
                              [0, sr, cr]])
    rotMatrixPitch = np.array([[cp, 0, sp],
                               [0, 1, 0],
                               [-sp, 0, cp]])
    rotMatrixYaw = np.array([[cy, -sy, 0],
                             [sy, cy, 0],
                             [0, 0, 1]])
    rotMatrix = np.matmul(rotMatrixYaw, np.matmul(rotMatrixPitch, rotMatrixRoll))
    """
    # More concise and faster implementation of the matrix derived above
    # However, still keeping the commented code above for reference
    rotMatrix = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])

    return np.matmul(rotMatrix, xyz)

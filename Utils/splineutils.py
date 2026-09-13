"""
Spline wrapper class
"""
# Python standard libraries
from typing import Literal

# External libraries
import scipy
import numpy as np

# Project python modules
from Utils.typealiases import Any, NDArrayFloat1D, NDArrayFloat2D, NDArrayNumber1D


class SplineND:
    """
    Multidimensional cubic B-spline wrapper with helper methods for multidimensional spline operations

    The spline is a scipy.interpolate.BSpline with degree 3 (k=3), created using scipy.interpolate.make_splprep()

    Note: A wrapper class is preferred for performance reasons as it allows pre-computing the (somewhat expensive)
    derivative and antiderivative splines, which are also used within the spline helper methods

    Methods:
        ...

    Attributes:
        ...
    """

    def __init__(self,
                 data: NDArrayFloat2D,
                 u: NDArrayFloat1D | None = None,
                 BCyclic: bool = False,
                 smoothing: float = 0,
                 uResolutionLUT: float = 1) -> None:
        """
        Initialise the SplineND object

        Args:
            TODO
        """
        # Dimension of the spline
        self.NDim = data.shape[1]

        # Calculate u parameterization if not provided, as the approximate displacement along the spline
        # Uses the linear distance between each data point corrected by approximated curvature between each data point
        if u is None:
            dataDiff = np.diff(data, axis=0)
            dot = (np.linalg.vecdot(dataDiff[:-1], dataDiff[1:])
                   / np.linalg.norm(dataDiff[:-1], axis=-1) / np.linalg.norm(dataDiff[1:], axis=-1))
            ADiff = np.acos(np.maximum(-1, np.minimum(dot, 1)))
            if BCyclic:
                ADiffBound = np.acos(np.dot(dataDiff[-1], dataDiff[0])
                                     / np.linalg.norm(dataDiff[-1]) / np.linalg.norm(dataDiff[0]))
                ADiff = (np.hstack([ADiffBound, ADiff]) + np.hstack([ADiff, ADiffBound])) / 2

            else:
                ADiff = (np.hstack([ADiff[0], ADiff]) + np.hstack([ADiff, ADiff[-1]])) / 2
            indsCurv = np.where(ADiff != 0)[0]
            rCorr = np.ones_like(ADiff)
            rCorr[indsCurv] = ADiff[indsCurv] / 2 / np.sin(ADiff[indsCurv] / 2)
            u = np.hstack([0, np.cumsum(rCorr * np.linalg.norm(dataDiff, axis=-1))])

        # Create spline to interpolate/fit the data
        self.spline, self.uData = scipy.interpolate.make_splprep(data.T, u=u, k=3, s=smoothing,
                                                                 bc_type='periodic' if BCyclic else 'not-a-knot')
        self.uMin = min(self.uData)
        self.uMax = max(self.uData)

        # Pre-compute spline derivatives
        self.splineDer = self.spline.derivative(nu=1)
        self.splineDer2 = self.spline.derivative(nu=2)

        # Pre-compute spline antiderivative - TODO: AntiDer isn't used i think?
        self.splineAntiDer = self.spline.antiderivative(nu=1)

        # Pre-compute displacement LUT along the spline - TODO: Consider making this a 1D spline?
        self.uLUT = np.linspace(self.uMin, self.uMax, 1 + int(np.ceil((self.uMax - self.uMin) / uResolutionLUT)))
        self.sLUT = np.append(0, scipy.integrate.cumulative_trapezoid(
                              np.linalg.norm(self.calc_tangent(self.uLUT, BNormalise=False), axis=-1), self.uLUT))

    def evaluate(self,
                 u: float | list[float] | NDArrayFloat1D) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Evaluate the spline at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Spline point(s) at the u parameter value(s)

            For multiple parameter values, the spline points will be vertically stacked
        """
        return self.spline(u).T

    def evaluate_der(self,
                     u: float | list[float] | NDArrayFloat1D) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Evaluate the spline derivative at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Derivative of the spline point(s) with respect to the spline parameter u at the u parameter value(s)

            For multiple parameter values, the derivatives will be vertically stacked
        """
        return self.splineDer(u).T

    def evaluate_der2(self,
                     u: float | list[float] | NDArrayFloat1D) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Evaluate the spline second derivative at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Second derivative of the spline point(s) with respect to the spline parameter u at the u parameter value(s)

            For multiple parameter values, the derivatives will be vertically stacked
        """
        return self.splineDer2(u).T

    def evaluate_antider(self,
                         u: float | list[float] | NDArrayFloat1D) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Evaluate the spline antiderivative at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Antiderivative of the spline point(s) with respect to the spline parameter u at the u parameter value(s)

            For multiple parameter values, the antiderivatives will be vertically stacked
        """
        return self.splineAntiDer(u).T

    def calc_tangent(self,
                     u: float | list[float] | NDArrayFloat1D,
                     BNormalise: bool = True) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Calculate the tangent vector at parameter value(s) u

        Args:
            u: Parameter value(s)
            BNormalise: Whether to normalise the tangent vector to unit length

        Returns:
            Tangent vector(s) at the u parameter values(s) along the spline

            The tangent vector(s) will have unit length if BNormalise, otherwise the length of the returned vector will
            be equal to the speed of the spline

            For multiple parameter values, the tangent vectors will be vertically stacked
        """
        tangent = self.evaluate_der(u)
        return (tangent.T / np.linalg.norm(tangent, axis=-1)).T if BNormalise else tangent

    def calc_normal_lat(self,
                        u: float | list[float] | NDArrayFloat1D,
                        BNormalise: bool = True) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Calculate the left-pointing normal vector (on the lateral xy plane) at parameter value(s) u

        Args:
            u: Parameter value(s)
            BNormalise: Whether to normalise the normal vector to unit length

        Returns:
            Normal vector(s) at the u parameter values(s) along the spline, pointing to the left

            The normal vector(s) will have unit length if BNormalise, otherwise the length of the returned vector will
            be equal to the speed of the spline

            For multiple parameter values, the normal vectors will be vertically stacked
        """
        # Calculate using the first 2 dimensions of the tangent vector as [-yTangent, xTangent]
        tangentT = self.calc_tangent(u, BNormalise=BNormalise).T
        normalT = np.zeros_like(tangentT)
        normalT[0] = -tangentT[1]
        normalT[1] = tangentT[0]
        return normalT.T

    def calc_Ay(self,
                u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate the spline angle about its yaw-rotated y-axis (i.e. pitch angle) at parameter value(s) u

        The pitch angle is measured in radians from the xy plane, positive pointing downwards, with range -pi to pi

        Args:
            u: Parameter value(s)

        Returns:
            Pitch angle(s) in radians at the u parameter values(s) along the spline
        """
        if self.NDim < 3:
            # Pitch angle is always 0 in 2D
            return np.zeros_like(u) if isinstance(u, list) else u * 0
        tangentT = self.calc_tangent(u, BNormalise=False).T
        return -np.arctan2(tangentT[2], np.linalg.norm(tangentT, axis=0))

    def calc_Az(self,
                u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate the spline angle about the z-axis (i.e. yaw angle) at parameter value(s) u

        The yaw angle is measured in radians anticlockwise from the positive x-axis, wrapping every 2 pi radians

        Args:
            u: Parameter value(s)

        Returns:
            Yaw angle(s) in radians at the u parameter values(s) along the spline
        """
        tangentT = self.calc_tangent(u, BNormalise=False).T
        return np.arctan2(tangentT[1], tangentT[0])


    def calc_kVec(self,
                  u: float | list[float] | NDArrayFloat1D) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Calculate the curvature vector at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Curvature vector(s) at the u parameter values(s) along the spline

            For multiple parameter values, the curvature vectors will be vertically stacked
        """
        return (self.evaluate_der2(u).T /
                np.pow(np.linalg.norm(self.calc_tangent(u, BNormalise=False).T, axis=0), 2)).T


    def calc_kTotal(self,
                    u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate the curvature magnitude at parameter value(s) u

        Args:
            u: Parameter value(s)

        Returns:
            Curvature magnitude(s) at the u parameter values(s) along the spline
        """
        return np.linalg.norm(self.calc_kVec(u), axis=-1)

    def calc_kLat(self,
                  u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate signed curvature in the lateral direction, positive curving left following the right-hand rule

        Args:
            u: Parameter value(s)

        Returns:
            Signed lateral curvature(s) at the u parameter values(s) along the spline
        """
        kVecT = self.calc_kVec(u).T
        return (np.linalg.norm(kVecT[:2], axis=0)
                * np.sign(np.sum(kVecT[:2] * self.calc_normal_lat(u, BNormalise=False).T[:2], axis=0)))

    def calc_kVert(self,
                   u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate signed curvature in the vertical direction, positive curving up following the right-hand rule

        Args:
            u: Parameter value(s)

        Returns:
            Signed vertical curvature(s) at the u parameter values(s) along the spline
        """
        if self.NDim < 3:
            # Vertical curvature is always 0 in 2D
            return np.zeros_like(u) if isinstance(u, list) else u * 0
        return self.calc_kVec(u).T[2]

    def calc_s(self,
               u: float | list[float] | NDArrayFloat1D) -> float | NDArrayFloat1D:
        """
        Calculate displacement along the spline from uStart to uEnd, by linearly interpolating the pre-computed LUT

        Args:
            u: Parameter value(s)

        Returns:
            Displacement along the spline at the u parameter values(s) along the spline
        """
        return np.interp(u, self.uLUT, self.sLUT)

    def param_closest(self,
                      data: NDArrayFloat2D,
                      uSeed: float | None = None) -> NDArrayFloat1D:
        ...

    def param_normal(self,
                     data: NDArrayFloat2D,
                     uSeed: float | None = None) -> NDArrayFloat1D:
        ...

"""
Spline wrapper class
"""
# Python standard libraries
from typing import Literal

# External libraries
import scipy
import numpy as np

# Project python modules
import Utils.utils as utils
from Utils.typealiases import Any, NDArrayFloat1D, NDArrayFloat2D, NDArrayNumber1D


class SplineND:
    """
    Multidimensional cubic B-spline wrapper with helper methods for multidimensional spline operations

    The spline is a scipy.interpolate.BSpline with degree 3 (k=3), created using scipy.interpolate.make_splprep()

    Note: A wrapper class is preferred for performance reasons as it allows pre-computing spline properties - derivative
    spline and displacement LUT

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

        # Calculate u parameterization as the (approximate) displacement along the spline, if not provided
        # Calculated using circular segment length - refer to: https://en.wikipedia.org/wiki/Circular_segment#Formulae
        #   Arc length = Sector angle / Curvature
        #   Chord length = 2 * sin(Sector angle / 2) / Curvature
        #   Rearranging: Curvature = 2 * sin(Sector angle / 2) / Chord length
        #   Therefore: Arc length = Sector angle / 2 / sin(Sector angle / 2) * Chord length
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
        self.uMin: float = min(self.uData)
        self.uMax: float = max(self.uData)

        # Pre-compute spline derivatives
        self.splineDer = self.spline.derivative(nu=1)
        self.splineDer2 = self.spline.derivative(nu=2)

        # Pre-compute displacement LUT along the spline - LUT is preferred over spline for speed and size reasons
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

    def calc_tangentDer(self,
                        u: float | list[float] | NDArrayFloat1D,
                        BUnit: bool = True) -> NDArrayFloat1D | NDArrayFloat2D:
        """
        Calculate the derivative of the tangent vector at parameter value(s) u

        Args:
            u: Parameter value(s)
            BUnit: Whether to use a unit tangent vector for the derivative

        Returns:
            Derivative of the tangent vector(s) at the u parameter values(s) along the spline

            The derivative of the tangent vector(s) will be for the unit tangent vector(s) is BUnit, otherwise the
            length of the returned vector will be equal to the acceleration of the spline

            For multiple parameter values, the derivatives of the tangent vectors will be vertically stacked
        """
        tangentDer = self.evaluate_der2(u)
        return (tangentDer.T / np.linalg.norm(self.calc_tangent(u, BNormalise=False), axis=-1)).T \
               if BUnit else tangentDer

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

        Definition of curvature reference: https://en.wikipedia.org/wiki/Curvature#General_parametrization

        Args:
            u: Parameter value(s)

        Returns:
            Curvature vector(s) at the u parameter values(s) along the spline

            For multiple parameter values, the curvature vectors will be vertically stacked
        """
        return (self.calc_tangentDer(u, BUnit=False).T
                / np.pow(np.linalg.norm(self.calc_tangent(u, BNormalise=False).T, axis=0), 2)).T

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

        Note that there is no internal validation that uMin <= u <= uMax, so the displacement returned will be as if u
        is clamped to be between uMin and uMax

        Args:
            u: Parameter value(s)

        Returns:
            Displacement along the spline at the u parameter values(s) along the spline
        """
        return np.interp(u, self.uLUT, self.sLUT)

    def param(self,
              data: NDArrayFloat2D,
              uSeed: float | None = None,
              BFast: bool = False) -> NDArrayFloat1D:
        """
        Parameterise data points by closest point on the spline

        By definition, this will parameterise the data points to be normal to the spline tangent

        This is a local optimisation starting at uSeed for the first data point, and subsequently using the
        parameterization of the previous point as the seed for the next data point

        Args:
            data: TODO
            uSeed: Initial u parameter candidate for the first data point, uses self.uMin if not provided
            BFast: Whether to skip the additional root-finding refinement of the parameterisation

        Returns:
            Parameterization of data points for closest point to the spline

        Raises:
            ValueError: Failed to converge for point {(i + 1)}/{len(data)} ({res.message})
        """
        if uSeed is None:
            uSeed = self.uMin

        def funjac_closest(u: NDArrayFloat1D,
                           point: NDArrayFloat1D) -> tuple[float, float]:
            """
            Calculate the objective function and its Jacobian for the first pass minimisation algorithm

            This is the distance of the spline at parameter u from the data point, and its derivative with respect to u

            Args:
                u: Candidate u parameter
                point: Data point

            Returns:
                Tuple of (obj, jac)

                obj: Objective function (distance of the spline at parameter u from the data point)

                jac: Jacobian of the objective function
            """
            u = utils.wrap(u[0], self.uMin, self.uMax)
            vec = self.evaluate(u) - point
            lVec = np.linalg.norm(vec)
            return lVec, np.dot(self.calc_tangent(u, BNormalise=False), vec / lVec)

        def fun_normal(u: NDArrayFloat1D,
                       point: NDArrayFloat1D) -> float:
            """
            Calculate the objective function for the second pass root-finding algorithm

            This is the cosine of the angle between the spline tangent and the vector from the spline at parameter u to
            the data point

            Args:
                u: Candidate u parameter
                point: Data point

            Returns:
                Objective function (cosine of the angle between the spline tangent and the vector from the spline at
                parameter u to the data point)
            """
            u = utils.wrap(u, self.uMin, self.uMax)
            vec = self.evaluate(u) - point
            return np.dot(self.calc_tangent(u), vec / np.linalg.norm(vec))

        # First pass - closest point minimisation
        method = 'BFGS'  # Slightly faster than 'Newton-CG', and much faster than other non-hessian methods
        fallbackMethod = 'Newton-CG' if method == 'BFGS' else 'BFGS'
        uParam = np.zeros(len(data))
        for i, point in enumerate(data):
            res = scipy.optimize.minimize(funjac_closest, uSeed, args=point, method=method, jac=True)
            if not res.success:
                # Fallback solver method
                res = scipy.optimize.minimize(funjac_closest, uSeed, args=point, method=fallbackMethod, jac=True)
                if not res.success:
                    raise ValueError(f"Failed to converge for point {(i + 1)}/{len(data)} ({res.message})")
            uSeed = res.x[0]
            uParam[i] = res.x[0]

        # Second pass - dot product root-finding
        if not BFast:
            method = 'secant'  # Around 2x faster than newton (no derivative provided)
            for i, point in enumerate(data):
                # TODO: provide the derivative (idk if it'll work bc every other time i've tried calculating has failed)
                sol = scipy.optimize.root_scalar(fun_normal, args=point, method=method, x0=uParam[i])
                if not sol.converged:
                    raise ValueError(f"Failed to converge for point {(i + 1)}/{len(data)} ({sol.flag})")
                uParam[i] = sol.root

        return uParam
import numpy as np
import pdb
import astropy.units as u
from astropy.coordinates import BaseRepresentation as vec

def _testing():
    print("To try random stuff out")

def KE(m, vel):
    """
    Computes kinteic energy of object
    Parameters
    ----------
    m: astropy.Quantity
        Mass of the object
    vel: astropy.coordinates.CartesianRepresentation
        3-D velocity of the object
    Returns
    -------
    ke: astropy.Quantity
        Kinetic energy
    """
    return 0.5*m*vel.norm()**2

def PE_point(M, dist):
    """
    Computes gravitational potential of
    a test object due to a point mass
    Parameters
    ----------
    M: astropy.Quantity
        Mass of the object
    dist: astropy.coordinates.CartesianRepresentation
        distance from the test mass
        to the point mass.
    Returns
    -------
    V: astropy.Quantity
        Graviatational potential
    """
    return -G*M/dist.norm()

def niceangle(angle,minval=0*u.rad,into='rad'):
    """
    Puts angle in the range
    $[0,2\pi]$.
    Parameters
    ----------
    angle: astropy.Quantity
        Angle in whatever units. Assumes
        radians if not given.
    minval: astropy.Quantity, optional
        Outputs angle in the range
        [minval, minval+2*pi] rad. 0 rad
        by default.
    into: str, optional
        Output can be given in degrees or 
        radians by setting this parameter
        to 'deg' or 'rad' respectively.
        Defaults to rad. If string is
        unrecognisable or not actually a
        string, outputs to radians but with
        a warning.
    """
    newangle = (angle.to(u.rad)-minval).value%(2*np.pi)*u.rad + minval
    if into is "deg":
        return newangle.to(u.deg)
    else:
        return newangle

def newton_rhapson(func,x0,deriv=None,tol=1e-15,maxiter=20):
    """
    An implementation of the newton-rhapson method of
    numerical root finding.
    Paramters
    ---------
    func: python callable function
        The function whose root is to be found
    x0: float
        Initial guess for root finding
    deriv: python callable function, optional
        The derivative of func. If not given,
        a first order numerical derivative
        is used. 
    tol: float, optional
        Tolerance for relative error between
        successive iterations
    maxiter: int, optional
        Maximum number of iterations this function
        is allowed to go through.
    Returns
    -------
    root: float
        Estimate of the root
    """
    #import pdb
    #pdb.set_trace()
    i = maxiter
    if deriv is None:
        deriv = lambda x: 1e15*(func(x+1e-15)-func(x))
    error = np.inf
    while (np.abs(error)>tol) and (i>0):
        dx =  -func(x0)/deriv(x0)
        error = dx/x0
        x0 = x0+dx
        i -=  1
    root = x0
    if i==0:
        raise UserWarning("Root not found to specified tolerance within {} iterations".format(maxiter))
        return root
    else:
        return root

    
import numpy as np
import pdb
import astropy.units as u
from astropy.units.astrophys import AU
from astropy.constants import G
from astropy.coordinates import CartesianRepresentation as cart
import helpers as hp

def _testing():
    print("To try random stuff out")


def cart2els(pos,vel,M,m,nat_units=False):
    """
    Converts cartesian coordinates to
    orbital elements assuming the cartesian
    coordinates are set in a frame that is
    centered on one of the objects.
    Parameters
    ----------
    pos: astropy.coordinates.CartesianRepresentation
        The position vector. The components must have
        units of distance.
    vel: astropy.coordinates.CartesianRepresentation
        The velocity vector. The components must have
        units of speed (distance/time).
    M: astropy.Quantity
        Mass of the object at (0,0).
    m: astropy.Quantity
        Mass of the object at the input
        coordinates.
    nat_units: bool, optional
        If true, returns answer in units where a = 1 and
        time is rescaled to sqrt(mu)t.

    Outputs
    -------
    els: astropy.Quantity list/array
        Returns an array of orbital elements
        for the osculating orbit.
    tau: astropy.Quantity 
        Some time parameter.
    """
    #import pdb
    #pdb.set_trace()
    mu = G*(M+m)
    R = pos.norm()
    V = vel.norm()
    a = 1/(2/R-V*V/mu)

    #Rescaling
    #t = t*np.sqrt(mu/a**3)
    pos = pos/a
    R = (R/a).decompose()
    vel = vel*np.sqrt(a/mu)
    V = (vel.norm()).decompose()
    #rDot = (pos.dot(vel)).decompose()/R
    
    #Rescaled angular momentum
    hvec = pos.cross(vel)
    h = (hvec.norm()).decompose()
    #pdb.set_trace()
    #Orbital elements
    e = np.sqrt(1-h*h)
    I = np.arccos(hvec.z/h)
    if I.value==0:
        Omega = 0*u.rad
        omegaPlusf = np.arccos(pos.x/R)
    else:
        Omega = np.arcsin(np.sign(hvec.z)*hvec.x/h/np.sin(I))
        omegaPlusf = np.arcsin(pos.z/R/np.sin(I))
    #because cos(Omega)= 1
    #pdb.set_trace()
    f = np.arccos(((1-e*e)/R-1)/e)
    omega = omegaPlusf - f
    
    f = hp.niceangle(f)
    I = hp.niceangle(I)
    Omega = hp.niceangle(Omega)
    omega = hp.niceangle(omega)

    #E = np.arccos((1-R)/e)
    #tau = t-E.value+e*np.sin(E)

    if nat_units:
        els = [1,e,I,Omega,omega,f]
    else:
        els = [a,e,I,Omega,omega,f] #(tau*np.sqrt(a**3/mu)).decompose()]
    return els

def els2cart(els,M,m):
    """
    The inverse function of cart2els.
    Converts orbital elements ot cartesian
    coordinates at the central mass
    Parameters
    ----------
    els: list of astropy Quantities
        Orbital elements in the same order
        as cart2els's output (with tau). If els
        is in natural units where a = 1, then
        the scaled positions and velocities are returned
    Returns
    -------
    pos: astropy.coordinates.CartesianRepresentation
        The position vector. The components must have
        units of distance.
    vel: astropy.coordinates.CartesianRepresentation
        The velocity vector. The components must have
        units of speed (distance/time).
    """
    #unpack els
    a,e,I,Omega,omega,f = els
    #tau = els[6]

    #Define constants
    mu = G*(M+m)

    if a == 1:
        scaled = True
    else:
        try:
            a.unit
            scaled = False
        except:
            raise TypeError("Please use astropy.Quantity objects to specify a's units or rescale to set a to 1 to use natural units.")
    
    #Work as if a is scaled and then rescale it
    #to standard units at the end.

    #Get h and its components
    #h = np.sqrt(1-e*e)
    #hz = h*np.cos(I)
    #hx = h*np.sign(hz)*np.sin(Omega)*np.sin(I)
    #hy = -h*np.sign(hz)*np.cos(Omega)*np.sin(I)
    #hvec = cart(x = hx,y = hy, z= hz)

    #Getting R and its components
    omegaPlusf = omega+f
    R = (1-e*e)/(e*np.cos(f)+1)
    z = R*np.sin(I)*np.sin(omegaPlusf)
    y = R*(np.cos(omegaPlusf)*np.sin(Omega)+np.sin(omegaPlusf)*np.cos(I)*np.cos(Omega))
    x = R*(np.cos(omegaPlusf)*np.cos(Omega)-np.sin(omegaPlusf)*np.cos(I)*np.sin(Omega))
    pos = cart(x = x,y = y, z = z)

    #Getting V and its components. I'm going to use Ruth's formula here.
    #n = 1 #In scaled units.
    xorbdot = -np.sin(f)/np.sqrt(1-e*e)
    yorbdot = (e+np.cos(f))/np.sqrt(1-e*e)
    vx = np.cos(Omega)*(xorbdot*np.cos(omega) - yorbdot*np.sin(omega)) - np.sin(Omega)* ((xorbdot*np.sin(omega) + yorbdot*np.cos(omega))*np.cos(I))   
    vy = np.sin(Omega)*(xorbdot*np.cos(omega) - yorbdot*np.sin(omega)) + np.cos(Omega)* ((xorbdot*np.sin(omega) + yorbdot*np.cos(omega))*np.cos(I))
    vz = (xorbdot*np.sin(omega) + yorbdot*np.cos(omega))*np.sin(I)
    vel = cart(x=vx,y=vy,z=vz)

    pos = a*pos
    vel = vel*(np.sqrt(mu/a)).to(u.km/u.s)
    return pos, vel

def helio2bary(pos_h,vel_h,M,m):
    """
    Converts position and velocity vectors in
    heliocentric frame to the barycentric frame.
    Parameters
    ----------
    pos_h: astropy.coordinates.CartesianRepresentation
        The position vector in the heliocentric frame.
        The components must have units of distance.
    vel_h: astropy.coordinates.CartesianRepresentation
        The velocity vector in the heliocentric frame.
        The components must have units of speed (distance/time).
    M: astropy.Quantity
        Central mass.
    m: astropy.Quantity
        Mass of the object at pos_h.
    Returns
    -------
    pos_b: astropy.coordinates.CartesianRepresentation
        Position in the barycentric frame
    vel_b: velocity in the barycentric frame
    """
    return pos_h*(M+m)/M, vel_h*(M+m)/M

def bary2helio(pos_h,vel_h,M,m):
    """
    Converts position and velocity vectors in
    heliocentric frame to the barycentric frame.
    Parameters
    ----------
    pos_h: astropy.coordinates.CartesianRepresentation
        The position vector in the heliocentric frame.
        The components must have units of distance.
    vel_h: astropy.coordinates.CartesianRepresentation
        The velocity vector in the heliocentric frame.
        The components must have units of speed (distance/time).
    M: astropy.Quantity
        Central mass.
    m: astropy.Quantity
        Mass of the object at pos_h.
    Returns
    -------
    pos_b: astropy.coordinates.CartesianRepresentation
        Position in the barycentric frame
    vel_b: velocity in the barycentric frame
    """
    return pos_h*M/(M+m), vel_h*M/(M+m)
    
def solve_kepler(M,e,E0=None,tol=1e-15,maxiter=20):
    """
    Solves the Kepler equation ($M = E - e\sinE$)
    for E given M and e using the Newton-Rhapson
    method (superconvergent).
    Parameters
    ----------
    M: float
        Value of n(t-t0). Must be dimensionless
    e: float
        Orbital eccentricity
    E0: float, optional
        Initial guess for E0. Defaults to M if
        not given
    tol: float, optional
        Tolerance for relative error between
        successive iterations
    maxiter: int, optional
        Maximum number of iterations this function
        is allowed to go through.
    Returns
    -------
    root: float
        Value of
    """
    if E0 is None:
        E0 = M
    func = lambda E: M-E+e*np.sin(E)
    deriv = lambda E: -1+e*np.cos(E)

    root = hp.newton_rhapson(func,E0,deriv,tol,maxiter)
    return root

def get_rad(t_array,M,m,a,e,t0=0*u.s):
    """
    Returns an array of radial distances
    at the instances in the input time
    array.
    Paramters
    ---------
    t_array: astropy.Quantity array
        An array of time instances at which
        the radial distances are to be found
    M: astropy.Quantity
        central mass
    m: astropy.Quantity
        Mass of the orbiting body
    a: astropy.Quantity
        Semi-major axis
    e: float
        Orbital eccentricity
    Returns
    -------
    radArray: astropy.Quantity array
        Array of radial distances.
    """
    #First, all computations are done in
    #natural units. Only while returning will the
    #answer be given in standard units
    #import pdb
    mu = G*(M+m)
    t_scaled = (t_array*np.sqrt(mu/a**3)).decompose().value
    t0_scaled = (t0*np.sqrt(mu/a**3)).decompose().value
    numT = len(t_array)
    radArray = np.zeros(numT)*AU

    for count, t in enumerate(t_scaled):
        Mval = t-t0_scaled
        E = solve_kepler(Mval,e)
        radArray[count] = a*(1-e*np.cos(E))
    
    return radArray



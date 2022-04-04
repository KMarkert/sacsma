import numpy as np
import numpy.typing as npt
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

from .date_utils import day_of_year

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit
def hargreaves(
    forcings: npt.NDArray[np.floating], 
    dates: npt.NDArray[np.datetime64]
) -> npt.NDArray[np.float64]:

    tmin = forcings[:,0]
    tmax = forcings[:,1]
    n = len(tmax)
    Gsc = 367
    lhov = 2.257

    doy = day_of_year(dates)

    tavg = np.mean(forcings,axis=1)

    eto = np.zeros_like(tavg)

    for i,t in enumerate(doy):
        b = 2 * np.pi * (t/365)
        Rav = 1.00011 + 0.034221*np.cos(b) + 0.00128*np.sin(b) + 0.000719*np.cos(2*b) + 0.000077*np.sin(2*b)
        Ho = ((Gsc * Rav) * 86400)/1e6

        eto[i] = (0.0023 * Ho * (tmax[i]-tmin[i])**0.5 * (tavg[i]+17.8))

    return eto

@jit
def hamon(forcings: npt.NDArray[np.floating], dates: npt.NDArray[np.datetime64], par: float, lat: float) -> npt.NDArray[np.float64]:

    tavg = np.mean(forcings,axis=0)
    n = len(tavg)

    doy = doy = day_of_year(dates)

    theta = 0.2163108 + 2 * np.atan(0.9671396 * np.tan(0.0086 * (doy - 186)))
    pi_v = np.asin(0.39795 * np.cos(theta))
    daylighthr = 24 - 24/np.pi * np.acos((np.sin(0.8333 * np.pi/180) + np.sin(lat * np.pi/180) * np.sin(pi_v))/(np.cos(lat * np.pi/180) * np.cos(pi_v)))

    esat = 0.611 * np.exp(17.27 * tavg/(237.3 + tavg))

    eto = par * 29.8 * daylighthr * (esat/(tavg + 273.2))

    return eto
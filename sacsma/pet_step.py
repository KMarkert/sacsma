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

    tmin = forcings[0]
    tmax = forcings[1]
    Gsc = 367
    lhov = 2.257

    doy = day_of_year(dates)

    tavg = np.mean(forcings)

    b = 2 * np.pi * (doy/365)
    Rav = 1.00011 + 0.034221*np.cos(b) + 0.00128*np.sin(b) + 0.000719*np.cos(2*b) + 0.000077*np.sin(2*b)
    Ho = ((Gsc * Rav) * 86400)/1e6

    eto = (0.0023 * Ho * (tmax-tmin)**0.5 * (tavg+17.8))

    return eto
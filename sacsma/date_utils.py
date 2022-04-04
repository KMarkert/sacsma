import numpy as np
import numpy.typing as npt
from numba import jit

@jit
def day_of_year(dates: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int16]:
    year = dates.astype("datetime64[Y]")
    doy = (dates - year) + 1
    return doy.astype(np.int16)
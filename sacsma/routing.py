import numpy as np
from scipy import stats
from scipy.integrate import quad
import numpy.typing as npt
from numba import jit

@jit
def lohmann(runoff: npt.NDArray[np.floating], baseflow: npt.NDArray[np.floating], flowlen: float, par: npt.ArrayLike) -> npt.NDArray[np.float64]:

    #------------------------------ Lohamann parameters -------------------------#
    N  = par[0]  # Grid Unit Hydrograph parameter (number of linear reservoirs)
    K  = par[1]  # Grid Unit Hydrograph parameter (reservoir storage constant)
    VELO = par[2]  # wave velocity in the linearized Saint-Venant equation(m/s)
    DIFF = par[3]  # diffusivity in the linearized Saint-Venant equation(m2/s)
    #----------------------------------------------------------------------------#

    #------- Base Time for HRU(watershed subunit) UH and channel routing UH -----#
    KE  = 12 
    UH_DAY = 96 
    DT = 3600        # Time step in second for solving Saint-Venant equation. This will affect TMAX
    TMAX = UH_DAY*24   # Base time of river routing UH in hour because DT is for an hour
    LE = 48*50
    #----------------------------------------------------------------------------#

    uh_river = np.zeros(UH_DAY)

    if flowlen == 0:
        uh_river[0] = 1

    else:

        t = 0 
        
        uhm_grid = np.zeros(LE)
        
        for k in range(LE):
        
            t = t + DT 
            pot = ((VELO*t-flowlen)**2)/(4*DIFF*t)

            if (pot <= 69):
                H = flowlen/(2 * t * np.sqrt(np.pi * t * DIFF)) * np.exp(-pot) 
            else:
                H = 0

            uhm_grid[k] = H 
        
        if np.sum(uhm_grid) == 0:
            uhm_grid[0] = 1
        
        else:
            uhm_grid = uhm_grid / np.sum(uhm_grid)


        FR = np.zeros((TMAX, 2))

        FR[0:24,1] = 1/24

        for t in range(TMAX):
            L = np.arange(1,TMAX+24+1)
            L = L[t-L > 0]

            FR[t, 1] = FR[t,1] + np.sum(FR[t-L,0] * uhm_grid[L])

        for i in range(1,UH_DAY+1):
            uh_river[i-1] = np.sum(FR[(24*i-23):(24*i),1])

    # HRU's Unit Hydrograph represented by Gamma distribution function

    uh_dist = stats.gamma(N, K)

    uh_hru_direct = np.zeros(KE)

    for i in range(KE):
        uh_hru_direct[i] = quad(uh_dist.pdf, 24*(i), 24*i+1)[0]

    uh_hru_base = np.zeros(KE)
    uh_hru_base[0] = 1


    # Combined UH for HRU's response at the watershed outlet 
    uh_direct = np.zeros(KE + UH_DAY - 1)
    uh_base = np.zeros(KE + UH_DAY - 1)

    for k in range(KE):
        for u in range(UH_DAY):
            uh_direct[k+u-1] = uh_direct[k+u-1] + uh_hru_direct[k] * uh_river[u] 
            uh_base[k+u-1]   = uh_base[k+u-1]   + uh_hru_base[k]   * uh_river[u] 

        
    uh_direct = uh_direct / np.sum(uh_direct)
    uh_base = uh_base / np.sum(uh_base)

    direct_flow = np.zeros_like(runoff)
    base_flow = np.zeros_like(baseflow)

    for i in range(len(direct_flow)):
    
        j = np.arange(0,(KE+UH_DAY-1))
        j = j[i-j+1 >= 1]

        direct_flow[i] = direct_flow[i] + np.sum(uh_direct[j] * runoff[i-j])
        base_flow[i] = base_flow[i] + np.sum(uh_base[j] * baseflow[i-j])

    return direct_flow, base_flow


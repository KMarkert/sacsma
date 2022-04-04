from cgitb import small
import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass


from .land_surface_step import sacsma
from .snow_step import snow17
from .pet_step import hargreaves
from .routing import lohmann
from .date_utils import day_of_year

@dataclass
class Simulation:
    forcings: pd.DataFrame
    elv: float
    snow_pars: npt.ArrayLike 
    ls_pars: npt.ArrayLike
    routing_pars:  npt.ArrayLike
    snow_state: npt.ArrayLike = np.array([0.,0.,0.,0.])
    ls_state: npt.ArrayLike = np.array([0.,0.,500.,500.,500.,0.])

    def __post_init__(self):

        self.n = self.forcings.shape[0]
        self.dates = self.forcings.index.values.astype(np.datetime64)

        self.runoffs = np.zeros((3,self.n),dtype=np.float64)
        self.sm = np.zeros(self.n,dtype=np.float64)
        self.we = np.zeros((2,self.n),dtype=np.float64)


    def execute(self):
        for i in range(self.n):
            self.step(i)

        flowlength = 71634.0

        direct,base = lohmann(self.runoffs[1,:], self.runoffs[2,:], flowlength, self.routing_pars)

        return direct + base

    def step(self, i):
        prcp = self.forcings["prcp"].iloc[i]
        tmin = self.forcings["tmin"].iloc[i]
        tmax = self.forcings["tmax"].iloc[i]
        tavg = np.mean([tmin, tmax])
        eto = hargreaves(np.array([tmin, tmax]) ,self.dates[i])
        snowmelt, self.snow_state = snow17(np.array([prcp, tavg]), self.dates[i], self.elv, self.snow_pars, self.snow_state)
        total_prcp = snowmelt + prcp
        self.runoffs[:,i], self.ls_state = sacsma(np.array([total_prcp, eto]), self.ls_pars, self.ls_state)

        self.sm[i] = self.ls_state[0]
        self.we[0,i] = self.snow_state[0]
        self.we[1,i] = self.snow_state[2]

        return
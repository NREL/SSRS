""" Module for computing a random thermal field - place-holder for Allen model 
    Thermal strength is a lognormal distribution, and terrain aspect is used to weight thermal
    formation toward south-facing (180) slopes
    Result is then smoothed with a Gaussian filter to form gaussian shaped thermals
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


def est_random_thermals(
    xsize: int,
    ysize: int,
    aspect: np.ndarray, #terrain aspect, used for weighting
    thermal_intensity_scale: float  #single parameter to describe strength of field
) -> np.ndarray:
    """ Returns field of smoothed random thermals from lognornal dist"""
    
    wt_init = np.zeros([ysize,xsize])
    border_x = int(0.05*xsize)
    border_y = int(0.05*ysize)
    for i in range(border_y,ysize-border_y):   #border with no thermals used to reduce problems of circling out of the domain
            for j in range(border_x,xsize-border_x):
                wtfactor=1000 + (abs(aspect[i,j]-180.)/180.)*2000.   #weight prob using aspect: asp near 180 has highest prob of a thermal
                num1=np.random.randint(1,int(wtfactor))
                if num1==5:
                    wt_init[i,j]=np.random.lognormal(thermal_intensity_scale+3, 0.5) 
                else:
                    wt_init[i,j]=0.0
        #        num1=np.random.randint(1,2000)   #est const = 2500 based on G Young 1.5 rule with 30 m grid

    
    wt=ndimage.gaussian_filter(wt_init, sigma=4, mode='constant')  #smooth the result to form Gaussian thermals
    
    return wt

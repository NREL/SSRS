""" Module defining possible actions for heuristics-based movements """
import numpy as np
#from scipy import ndimage #for smoothing updraft field
from scipy.interpolate import RectBivariateSpline


def random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                step=100.0,halfsector=90.0):
    """Perform a random movement, relative to the previous direction of
    movement, neglecting the PAM

    Notes:
    - Currently, only information from the previous position is used.

    Parameters
    ----------
    trajectory: list
        All previous positions [m] along the track
    directions: list
        Vectors defining the movement for all track segments
    PAM: float
        Principle axis of migration, clockwise from north [deg]
    wo_interp: function
        Orographic updraft field w_o(x,y) that can be evaluated at an
        arbitrary location
    step: float
        Distance [m] to move in one step
    halfsector: float
    
    """
    cur_pos = trajectory[-1]
    last_dir = directions[-1]
    ref_ang = np.arctan2(last_dir[1],last_dir[0]) # previous movement angle (counterclockwise from +x)                       
    ang0 = np.radians(ref_ang - halfsector)
    ang1 = np.radians(ref_ang + halfsector)
    
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    
    return cur_pos + delta


def dir_random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                step=100.0,halfsector=45.0):
    cur_pos = trajectory[-1]
    """Perform a random movement along the principle axis of migration (PAM)

    Parameters
    ----------
    trajectory: list
        All previous positions [m] along the track
    directions: list
        Vectors defining the movement for all track segments
    PAM: float
        Principle axis of migration, clockwise from north [deg]
    wo_interp: function
        Orographic updraft field w_o(x,y) that can be evaluated at an
        arbitrary location
    step: float
        Distance [m] to move in one step
    halfsector: float
        New movement direction will be within PAM +/- halfsector [deg]
    """
    ang0 = np.radians((90. - PAM) - halfsector)
    ang1 = np.radians((90. - PAM) + halfsector)
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    return cur_pos + delta

def step_ahead_drw(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                step=100.0,dist=100.0,halfsector=30,Nsearch=10,threshold=0.85,sigma=0.0):
    """Perform a step forward in near-PAM direction based on nearby updraft values

    Notes
    - Currently, only information from the previous position is used.

    """
    cur_pos = trajectory[-1]      
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
                
    w_max,idxmax,best_dir,elev_w_max=searcharc_wo(trajectory,directions,PAM,wo_interp,elev_interp,
                    step=100.0,dist=100.0,halfsector=30.0,Nsearch=10)
    
    # take step based on search results, not allowing movement steeply downslope
    if w_max > threshold and (elev_w_max - elev_cur_pos)>-20:
        if sigma > 0:
            # add additional uncertainty
            best_dir = np.random.normal(best_dir, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    elif w_max < threshold and w_max >= 0.75*threshold and (elev_w_max - elev_cur_pos)>-20:
        #lift is near threshold so choose in that direction +-
        best_dir = np.random.normal(best_dir, scale=np.radians(30))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    else:
        # no usable updraft found adjacent, do a directed random walk
        new_pos = dir_random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                              step=step,halfsector=halfsector)
    return new_pos


def step_ahead_look_ahead(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                    step=100.0,dist=100.0,halfsector=30,Nsearch=10,threshold=0.85,sigma=0.0):
    """Perform a step forward in near-PAM direction based on nearby updraft values

    Notes:
    - Currently, only information from the previous position is used.

    """
    cur_pos = trajectory[-1]      
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
                
    w_max,idxmax,best_dir,elev_w_max=searcharc_wo(trajectory,directions,PAM,wo_interp,elev_interp,
        step=100.0,dist=100.0,halfsector=30.0,Nsearch=10)
    
    # take step based on search results, not allowing movement steeply downslope
    if w_max > threshold and (elev_w_max - elev_cur_pos)>-20:
        if sigma > 0:
            # add additional uncertainty
            best_dir = np.random.normal(best_dir, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    elif w_max < threshold and w_max >= 0.75*threshold and (elev_w_max - elev_cur_pos)>-20:
        #lift is near threshold so choose in that direction +-
        best_dir = np.random.normal(best_dir, scale=np.radians(30))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    else:
        # no usable updraft found adjacent, look ahead for strong updraft region
        new_pos = look_ahead_v2(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                            step=step,dist=2000.0,halfsector=45,Nsearch=10,threshold=threshold,sigma=0.0)
    return new_pos


def look_ahead(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
               step=100.0,dist=100.0,halfsector=45.0,Nsearch=5,threshold=0.85,sigma=0.0):
    """Perform a movement based on some knowledge of the flowfield ahead
    searching along an arc of radius dist

    Notes:
    - Currently, only information from the previous position is used.

    Parameters
    ----------
    trajectory: list
        All previous positions [m] along the track
    directions: list
        Vectors defining the movement for all track segments
    PAM: float
        Principle axis of migration, clockwise from north [deg]
    wo_interp: function
        Orographic updraft field w_o(x,y) that can be evaluated at an
        arbitrary location
    step: float
        Distance [m] to move in one step
    dist: float
        Search distance [m] to look ahead
    halfsector: float
        Search points are along an arc, spanning the previous direction
        of movement +/- halfsector [deg]
    Nsearch: int
        Number of search points along the lookahead arc
    threshold: float
        Updraft strength [m/s] for which the movement is influenced
    sigma: float
        Uncertainty [deg] in the resulting movement direction, if the
        updraft threshold is exceeded in the lookahead arc
    """
    
    cur_pos = trajectory[-1]
    last_dir = directions[-1]
    ref_ang = np.arctan2(last_dir[1],last_dir[0]) # previous movement angle (counterclockwise from +x)
    # search for usable updrafts
    ang0 = ref_ang - np.radians(halfsector)
    ang1 = ref_ang + np.radians(halfsector)
    search_arc = np.linspace(ang0, ang1, Nsearch)
    search_x = cur_pos[0] + dist * np.cos(search_arc)
    search_y = cur_pos[1] + dist * np.sin(search_arc)
    check_w = wo_interp(search_x, search_y, grid=False)
    w_max = np.max(check_w)
    # decide what to do with search information
    if w_max > threshold:
        # go towards max updraft
        idxmax = np.argmax(w_max)
        best_dir = search_arc[idxmax]
        if sigma > 0:
            # add additional uncertainty
            best_dir = np.random.normal(best_dir, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    else:
        # no usable updraft found, do a random walk
        new_pos = dir_random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,
                              step=step,halfsector=halfsector)
    return new_pos


def look_ahead_v2(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                  step=100.0,dist=1000.0,rangedelta=250.0,halfsector=45.0,
                  Nsearch=5,threshold=0.85,sigma=0.0):
    """Perform a movement based on some knowledge of the _smoothed_
    flowfield ahead, searching within a sector out to radius `dist` in
    radial increments of `rangedelta`

    Notes:
    - Currently, only information from the previous position is used.
    """
    cur_pos = trajectory[-1]
                     
    num = int(dist/rangedelta)
    assert num > 0, f'dist={dist} should be greater than rangedelta={rangedelta}'
    w_best = np.zeros(num,dtype=float)
    loc_best = np.zeros(num,dtype=float)
    dir_best = np.zeros(num,dtype=float)
    z_w_best = np.zeros(num,dtype=float)
    rad_dist = np.zeros(num,dtype=float)

    for k,radius in enumerate(np.arange(rangedelta, dist, rangedelta)):
        # search sector of smoothed wo in successive arcs with rangedelta
        # increment in radius
        numsearchlocs = int(radius/25.)
        w_max,idxmax,best_dir,elev_w_max = searcharc_wo(
                        trajectory,directions,PAM,wo_sm_interp,elev_interp,
                        step=step,dist=radius,halfsector=halfsector,Nsearch=numsearchlocs)
        w_best[k] = w_max
        loc_best[k] = idxmax
        dir_best[k] = best_dir
        z_w_best[k] = elev_w_max
        rad_dist[k] = radius

    # find max w over all arcs
    w_best_sorted = -np.sort(-w_best)
    loc_best_sorted = w_best.argsort()[-2:][::-1]
    w_globalmax = w_best_sorted[0]
    idxglobalmax = loc_best_sorted[0]
    elev_w_globalmax = z_w_best[idxglobalmax]
    elev_cur_pos = elev_interp(cur_pos[0],cur_pos[1], grid=False)
    
    # TODO: maybe what we want is to get the two best, and set the probability of one or the other based on the inverse dist to cur pos
    
    # take step towards max updraft 
    if w_globalmax > threshold and elev_w_globalmax >= 0.75*elev_cur_pos:
        best_dir = dir_best[idxglobalmax]
        #if sigma > 0:
            # add additional uncertainty
        #    best_dir = np.random.normal(best_dir, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    else:
        # no usable updraft found, do a random walk, either directed or not
        randwalk=np.random.randint(1, 10)  #set frequency for each type of movement
        if randwalk==1:  #10% of the time take a totally random walk
            # TODO: implement number of steps > 1
            #randy2=np.random.randint(10, 30) #number of steps - ask Eliot
            new_pos=random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                step=100.0,halfsector=90.0)
        else:
            new_pos = dir_random_walk(trajectory,directions,PAM,wo_interp,wo_sm_interp,elev_interp,
                              step=step,halfsector=halfsector)
    return new_pos
    
    
def searcharc_wo(trajectory,directions,PAM,wo_interp,elev_interp,
                    step=100.0,dist=100.0,halfsector=45.0,Nsearch=10):
    
    cur_pos = trajectory[-1]
    
    ang0 = np.radians(90-PAM-halfsector)
    ang1 = np.radians(90-PAM+halfsector)
    search_arc = np.linspace(ang0, ang1, Nsearch)
    search_x = cur_pos[0] + dist * np.cos(search_arc)
    search_y = cur_pos[1] + dist * np.sin(search_arc)
    check_w = wo_interp(search_x, search_y, grid=False)
    check_elev=elev_interp(search_x,search_y, grid=False)
    
    w_sorted=-np.sort(-check_w)
    id_wsorted=check_w.argsort()[-3:][::-1]   #return top three values
    w_max1=w_sorted[0]
    idxmax1=id_wsorted[0]
    w_max2=w_sorted[1]
    idxmax2=id_wsorted[1]
    best_dir1 = search_arc[idxmax1]
    best_dir2 = search_arc[idxmax2]
    xcoord_wmax1 = cur_pos[0] + dist * np.cos(search_arc[idxmax1])
    ycoord_wmax1 = cur_pos[1] + dist * np.sin(search_arc[idxmax1])
    elev_w_max1 = elev_interp(xcoord_wmax1,ycoord_wmax1, grid=False)

    return w_max1,idxmax1,best_dir1,elev_w_max1

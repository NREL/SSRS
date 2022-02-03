""" Module defining possible actions for heuristics-based movements """
import numpy as np

def random_walk(trajectory,directions,PAM,wo_interp,step=100.0,halfsector=45.0):
    cur_pos = trajectory[-1]
    """Perform a random movement along the principle axis of migration (PAM)

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
        New movement direction will be within PAM +/- halfsector [deg]
    """
    ang0 = np.radians((90. - PAM) - halfsector)
    ang1 = np.radians((90. - PAM) + halfsector)
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    return cur_pos + delta

def look_ahead(trajectory,directions,PAM,wo_interp,step=100.0,
               dist=100.0,halfsector=45.0,Nsearch=5,threshold=0.85,sigma=0.0):
    """Perform a movement based on some knowledge of the flowfield ahead

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
        new_pos = random_walk(trajectory,directions,PAM,wo_interp,
                              step=step,halfsector=halfsector)
    return new_pos


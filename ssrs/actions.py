""" Module defining possible actions for heuristics-based movements """
import numpy as np

def random_walk(trajectory,directions,PAM,wo_interp,step=100.0,sector=45.0):
    cur_pos = trajectory[-1]
    ang0 = np.radians((90. - PAM) - sector)
    ang1 = np.radians((90. - PAM) + sector)
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    return cur_pos + delta

def look_ahead(trajectory,directions,PAM,wo_interp,step=100.0,
               dist=100.0,sector=45.0,Nsearch=5,threshold=0.85):
    cur_pos = trajectory[-1]
    last_dir = directions[-1]
    ref_ang = np.arctan2(last_dir[1],last_dir[0])
    ang0 = ref_ang - np.radians(sector)
    ang1 = ref_ang + np.radians(sector)
    search_arc = np.linspace(ang0, ang1, Nsearch)
    search_x = cur_pos[0] + dist * np.cos(search_arc)
    search_y = cur_pos[1] + dist * np.sin(search_arc)
    check_w = wo_interp(search_x, search_y, grid=False)
    w_max = np.max(check_w)
    if w_max > threshold:
        # go towards max updraft
        idxmax = np.argmax(w_max)
        best_dir = search_arc[idxmax]
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
    else:
        # no usable updraft found, do a random walk
        new_pos = random_walk(trajectory,directions,PAM,wo_interp,
                              step=step,sector=sector)
    return new_pos


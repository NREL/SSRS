""" Module defining possible actions for heuristics-based movements """
import numpy as np

def random_walk(trajectory,directions,PAM,wo_interp,step=100.0,sector=45.0):
    cur_pos = trajectory[-1]
    ang0 = np.radians((90. - PAM) - sector)
    ang1 = np.radians((90. - PAM) + sector)
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    return cur_pos + delta


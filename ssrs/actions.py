"""Module defining possible actions for heuristics-based movements 

An action function is defined generally as:

    action(*args, **kwargs)

*args are common data passed from
`movmodel.generate_heuristic_eagle_track()` to all action functions, providing
information about the eagle movement history and environment, which may or may
not be used. **kwargs are optional, action-specific keyword arguments.

The default arguments (*args) include:

* trajectory: list
    The history of all previous positions [m] along the track, including
    the current position
* directions: list
    Vectors defining the direction of movement for all track segments
* track_weight: list
    Risk weighting factor based on presumed altitude for computing presence map: low (wt=1), moderate (wt=0.5), or high flight (wt=0)
* PAM: float
    Principle axis of migration, clockwise from north [deg]
* windspeed: float
* winddir: float
* wo_interp: function
    Orographic updraft field w_o(x,y) that can be evaluated at an
    arbitrary location [m/s]
* wo_sm_interp: function
    Smoothed orographic updraft field w_o(x,y) that can be evaluated at
    an arbitrary location [m/s]
* wt_interp: function
    Thermal updraft field w_t(x,y) that can be evaluated at an arbitrary
    location [m/s]
* elev_interp: function
    Elevation z(x,y) that can be evaluated at an arbitrary location [m]

`kwargs` is an optional list of keywords describing additional action-specific
parameters.

"""
import numpy as np
#from scipy import ndimage #for smoothing updraft field
from scipy.interpolate import RectBivariateSpline


def random_walk(trajectory,directions,track_weight,PAM,windspeed,winddir,theshold,lookaheaddist,maxx,maxy,
                wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=90.0):
    """Perform a random movement, neglecting the PAM

    Notes:
    - Currently, only information from the previous position is used.

    Additional Parameters
    ---------------------
    step: float
        Distance [m] to move in one step
    halfsector: float
        New movement direction will be within
          [prevdir-halfsector, prevdir+halfsector],
        where prevdir is the previous direction of movement [deg]
    """
    cur_pos = trajectory[-1]
    last_dir = directions[-1]
    ref_ang = np.arctan2(last_dir[1],last_dir[0]) # previous movement angle (counterclockwise from +x)                       
    ang0 = ref_ang - np.radians(halfsector)
    ang1 = ref_ang + np.radians(halfsector)
    
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    step_wt=1. #typically a random movement might be to interact with another eagle or search for prey, etc
    
    return cur_pos + delta,step_wt


def dir_random_walk(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=15.0):
    cur_pos = trajectory[-1]
    """Perform a random movement along the principle axis of migration (PAM)

    Additional Parameters
    ---------------------
    step: float
        Distance [m] to move in one step
    halfsector: float
        New movement direction will be within PAM +/- halfsector [deg]
    """
    ang0 = np.radians((90. - PAM) - halfsector)
    ang1 = np.radians((90. - PAM) + halfsector)
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    step_wt=0.5 #assume a moderate height for a directed movement (powered flight, glide, etc)
    
    return cur_pos + delta,step_wt

def step_ahead_drw(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,
                maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,
                step=50.0,dist=50.0,halfsector=30,Nsearch=10,sigma=0.0):
    """Perform a step forward in near-PAM direction based on nearby updraft values
        If thermal lift is detected, go to thermal soar and glide movement
        If no updraft above threshold, do a directed random walk step
    """
    cur_pos = trajectory[-1]  
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
                
    wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                    trajectory,directions,track_weight,PAM,wo_interp,wt_interp,elev_interp,
                    step=step,dist=dist,halfsector=halfsector,Nsearch=Nsearch)
    
    # take step based on search results, preferring movement upslope
    # if wt exceeds wo, do a thermal-gliding action
    if (wt_max1 > wo_max1) & (wt_max1 > threshold):
        delta = step * np.array([np.cos(best_dir_wt1),np.sin(best_dir_wt1)])
        new_pos = cur_pos + delta
        directions.append(delta)
        step_wt=0.5  #entering thermal
        trajectory.append(new_pos)
        track_weight.append(step_wt)
        new_pos,step_wt = thermal_soar_glide(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                        wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0,
        uniform_winddirn=uniform_winddirn)
    
    elif wo_max1 > threshold and (elev_wo_max1 > elev_wo_max2):
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max2 > threshold and (elev_wo_max2 > elev_wo_max1):
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo2 = np.random.normal(best_dir_wo2, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo2),np.sin(best_dir_wo2)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max1 > threshold and (elev_wo_max1 - elev_cur_pos)>-20:
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max1 < threshold and wo_max1 >= 0.75*threshold and (elev_wo_max1 - elev_cur_pos)>0:
        #lift is near threshold so choose in that direction +-
        best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(30))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    else:
        # no usable updraft found adjacent, do a directed random walk step
        new_pos,step_wt = dir_random_walk(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                                  wo_interp,wo_sm_interp,wt_interp,elev_interp,step=step,halfsector=15.0)
    return new_pos,step_wt


def step_ahead_look_ahead(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                    wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,dist=50.0,halfsector=30,Nsearch=10,sigma=0.0):
    """Perform a step forward in near-PAM direction based on nearby updraft values, but
        If thermal lift is detected, go to thermal soar and glide movement
        If no useable updraft nearby, go to the look-ahead movement
                    
    """
    cur_pos = trajectory[-1]    
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
                
    wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
            trajectory,directions,track_weight,PAM,wo_interp,wt_interp,elev_interp,
            step=step,dist=dist,halfsector=halfsector,Nsearch=Nsearch)
    
    # take step based on search results, preferring movement upslope
    # if wt exceeds wo, do a thermal-gliding action
    if (wt_max1 > wo_max1) & (wt_max1 > threshold):
        delta = step * np.array([np.cos(best_dir_wt1),np.sin(best_dir_wt1)])
        new_pos = cur_pos + delta
        directions.append(delta)
        step_wt=0.5  #entering thermal
        trajectory.append(new_pos)
        track_weight.append(step_wt)
        new_pos,step_wt=thermal_soar_glide(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                    wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0)
    
    elif wo_max1 > threshold and (elev_wo_max1 > elev_wo_max2):
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max2 > threshold and (elev_wo_max2 > elev_wo_max1):
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo2 = np.random.normal(best_dir_wo2, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo2),np.sin(best_dir_wo2)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max1 > threshold and (elev_wo_max1 - elev_cur_pos)>-20:
        if sigma > 0:
            # add additional uncertainty
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wo_max1 < threshold and wo_max1 >= 0.75*threshold and (elev_wo_max1 - elev_cur_pos)>0:
        #lift is near threshold so choose in that direction +-
        best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(30))
        delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    else:
        # no usable updraft found adjacent, look ahead for strong updraft region
        # TODO: should we add secondary search parameters as kwargs?
        new_pos,step_wt = look_ahead(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                        wo_interp,wo_sm_interp,wt_interp,elev_interp,step=step,halfsector=45,Nsearch=10,sigma=0.0)
    
    return new_pos,step_wt

def look_ahead(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,rangedelta=250.0,halfsector=45.0,Nsearch=5,sigma=0.0):
    """Perform a movement based on sampling the updraft field ahead,
    searching within a sector out to radius `dist` in
    radial increments of `rangedelta`
    TODO: Replace legacy look_ahead function with this 
    """
    cur_pos = trajectory[-1]
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
           
    num = int(lookaheaddist/rangedelta)
    assert num > 0, f'look_ahead_distance={look_ahead_dist} should be greater than rangedelta={rangedelta}'
    wo_best = np.zeros(num,dtype=float)
    loc_wobest = np.zeros(num,dtype=float)
    dir_wobest = np.zeros(num,dtype=float)
    wt_best = np.zeros(num,dtype=float)
    loc_wtbest = np.zeros(num,dtype=float)
    dir_wtbest = np.zeros(num,dtype=float)    
    elev_wobest = np.zeros(num,dtype=float) 
    rad_dist = np.zeros(num,dtype=float)

    for k,radius in enumerate(np.arange(rangedelta, lookaheaddist, rangedelta)):
        # search sector of smoothed wo and wt in successive arcs with rangedelta
        # increment in radius
        numsearchlocs = int(radius/25.)
        wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                        trajectory,directions,track_weight,PAM,wo_sm_interp,wt_interp,elev_interp,
                        step=step,dist=radius,halfsector=halfsector,Nsearch=numsearchlocs)
        wo_best[k] = wo_max1
        wt_best[k] = wt_max1
        loc_wobest[k] = idx_womax1
        loc_wtbest[k] = idx_wtmax1
        dir_wobest[k] = best_dir_wo1
        dir_wtbest[k] = best_dir_wt1  
        elev_wobest[k] = elev_wo_max1 
        rad_dist[k] = radius

    # find max wo and max wt over all search arcs
    wo_best_sorted = -np.sort(-wo_best)
    loc_bestwo_sorted = wo_best.argsort()[-2:][::-1]
    wo_globalmax = wo_best_sorted[0]
    idx_wo_globalmax = loc_bestwo_sorted[0]
    bestdir_wo = dir_wobest[idx_wo_globalmax]
    wo_best_elev = elev_wobest[idx_wo_globalmax]
    
    wt_best_sorted = -np.sort(-wt_best)
    loc_bestwt_sorted = wt_best.argsort()[-2:][::-1]
    wt_globalmax = wt_best_sorted[0]
    idx_wt_globalmax = loc_bestwt_sorted[0]
    bestdir_wt = dir_wtbest[idx_wt_globalmax]
    
    # take step towards max wo or towards max wt based on least dist from cur_pos
    # w must be well above threshold for bird to invest energy to move that dir
    # and location cannot be at signif lower altitude
    if bestdir_wo == bestdir_wt:
        best_dir = bestdir_wo
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
        step_wt=1.    
    elif wo_globalmax > 2*threshold and (wo_best_elev - elev_cur_pos)>-25 and idx_wo_globalmax < idx_wt_globalmax:
        #best_dir = bestdir_wo
        best_dir= np.random.normal(bestdir_wo, scale=np.radians(15))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wt_globalmax > 2*threshold and idx_wt_globalmax < idx_wo_globalmax:
        #best_dir = bestdir_wt
        best_dir= np.random.normal(bestdir_wt, scale=np.radians(15))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
        step_wt=0.5   #entering thermal
    elif wo_globalmax > 2*threshold and (wo_best_elev - elev_cur_pos)>-25 and wt_globalmax < 2*threshold:
        #best_dir = bestdir_wo
        best_dir= np.random.normal(bestdir_wo, scale=np.radians(15))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
        step_wt=1.  #using orographic lift
    elif wt_globalmax < 2*threshold and wt_globalmax > 2*threshold:
        #best_dir = bestdir_wt
        best_dir= np.random.normal(bestdir_wt, scale=np.radians(15))
        delta = step * np.array([np.cos(best_dir),np.sin(best_dir)])
        new_pos = cur_pos + delta
        step_wt=0.5  #entering thermal 
    else:
        # no usable updraft found, do a dir random walk
        randy2=np.random.randint(5, 10) #number of steps
        #new_pos = []
        #step_wt=[]
        for _ in range(randy2):
            new_pos,step_wt =dir_random_walk(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,
                maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,step=step,halfsector=15.0)
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)    
               
    return new_pos,step_wt
    
    
def searcharc_w(trajectory,directions,track_weight,PAM,wo_interp,wt_interp,elev_interp,
                    step=50.0,dist=50.0,halfsector=45.0,Nsearch=10):
    """Perform a search for lift along a sector of 90 deg at a radial dist = dist
    """
    cur_pos= trajectory[-1]
    
    ang0 = np.radians(90-PAM-halfsector)
    ang1 = np.radians(90-PAM+halfsector)
    search_arc = np.linspace(ang0, ang1, Nsearch)
    search_x = cur_pos[0] + dist * np.cos(search_arc)
    search_y = cur_pos[1] + dist * np.sin(search_arc)
    check_wo = wo_interp(search_x, search_y, grid=False)
    check_wt = wt_interp(search_x, search_y, grid=False)
    check_wtot=check_wo+check_wt
    check_elev=elev_interp(search_x,search_y, grid=False)
    
    wo_sorted=-np.sort(-check_wo)
    id_wosorted=check_wo.argsort()[-2:][::-1]   #return top two wo values of wo
    wo_max1=wo_sorted[0]
    idx_womax1=id_wosorted[0]
    wo_max2=wo_sorted[1]
    idx_womax2=id_wosorted[1]
    
    wt_sorted=-np.sort(-check_wt)
    id_wtsorted=check_wt.argsort()[-2:][::-1]   #return top wt value
    wt_max1=wt_sorted[0]
    idx_wtmax1=id_wtsorted[0]
    
    wtot_sorted=-np.sort(-check_wtot)
    id_wtotsorted=check_wtot.argsort()[-2:][::-1]   #not using this currently...
    wtot_max1=wt_sorted[0]
    idx_wtotmax1=id_wtotsorted[0]
    
    elev_sorted=-np.sort(-check_elev)
    id_elevsorted=check_elev.argsort()[-2:][::-1]   
    elev_max=elev_sorted[0]
    idx_elevmax=id_elevsorted[0]
   
    
    best_dir_wo1 = search_arc[idx_womax1]  #best dirs for womax1 womax2 wtmx1 elevmax
    best_dir_wo2 = search_arc[idx_womax2]
    best_dir_wt1 = search_arc[idx_wtmax1]
    best_dir_elev = search_arc[idx_elevmax]
    
    xcoord_womax1 = cur_pos[0] + dist * np.cos(search_arc[idx_womax1])
    ycoord_womax1 = cur_pos[1] + dist * np.sin(search_arc[idx_womax1])
    elev_wo_max1 = elev_interp(xcoord_womax1,ycoord_womax1, grid=False)  #elev at wo max
    
    xcoord_womax2 = cur_pos[0] + dist * np.cos(search_arc[idx_womax2])
    ycoord_womax2 = cur_pos[1] + dist * np.sin(search_arc[idx_womax2])
    elev_wo_max2 = elev_interp(xcoord_womax2,ycoord_womax2, grid=False)
    
    xcoord_elevmax = cur_pos[0] + dist * np.cos(search_arc[idx_elevmax])
    ycoord_elevmax = cur_pos[1] + dist * np.sin(search_arc[idx_elevmax])
    wo_elevmax = wo_interp(xcoord_elevmax,ycoord_elevmax, grid=False)   #wo at elev max
    

    return wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2


def thermal_soar_glide(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                        wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0):
    """Three part movement is taken whenever thermal lift > threshold is encountered
            1. search for center of thermal using the searcharc() function
            2. circle and drift downwind for specified number of times (4 to 12 times)
            3. glide in the PAMward direction for a distance proportional to circling time             
    """                    
    cur_pos = trajectory[-1]
    last_dir = directions[-1]
        
    #first, search outward arcs of 50 m out to 300 m to find center of thermal
    wt_best = np.zeros(6,dtype=float)
    loc_best = np.zeros(6,dtype=float)
    dir_best = np.zeros(6,dtype=float)
    rad_dist = np.zeros(6,dtype=float)
    for k,radius in enumerate(np.arange(50, 300.1, 50)):
        # search sector of wt in successive arcs to 300 m
        numsearchlocs = int(radius/5.)
        wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                        trajectory,directions,track_weight,PAM,wo_sm_interp,wt_interp,elev_interp,
                        step=step,dist=radius,halfsector=halfsector,Nsearch=numsearchlocs)
        wt_best[k] = wt_max1
        loc_best[k] = idx_wtmax1
        dir_best[k] = best_dir_wt1
        rad_dist[k] = radius

    # now, find max wt over all arcs and move there
    wt_best_sorted = -np.sort(-wt_best)
    loc_best_sorted = wt_best.argsort()[-2:][::-1]
    wt_globalmax = wt_best_sorted[0]
    idxglobalmax = loc_best_sorted[0]
    best_dir_wt = dir_best[idxglobalmax]
    radius_wt=rad_dist[idxglobalmax]
    
    delta = radius_wt * np.array([np.cos(best_dir_wt),np.sin(best_dir_wt)])
    new_pos = cur_pos + radius_wt * np.array([np.cos(best_dir_wt),np.sin(best_dir_wt)]) #location of max wt
    
    directions.append(delta)
    step_wt=0.5  #base of thermal, assume moderate height
    trajectory.append(new_pos)
    track_weight.append(step_wt)
    
    last_dir = delta
    ref_ang = np.arctan2(last_dir[1],last_dir[0]) # previous movement angle (counterclockwise from +x)  
    cur_pos=new_pos
    
    #circle up and drift downwind in thermal
    
    #assume minimum of 4 loops (32 steps) and max of 20 loops (160 steps), proportional to wt of the thermal, with max of 8 m/s
    #note that it takes 8 steps for a full circle with angle = pi/4
    circlesteps=int(32+((wt_globalmax-threshold)/(8-threshold))*(160-32))
    #circlesteps=np.random.randint(32, 96) #number of steps circling
    soar_rad=np.random.uniform(30, 100) #soaring radius

    #TODO this assumes a constant windfield everywhere. Need to code for WTK with u(x,y) and v(x,y)
    uwind=windspeed*np.cos(np.radians(270.-winddir))
    vwind=windspeed*np.sin(np.radians(270.-winddir))

    dir=np.random.randint(1,3) #direction of soar, either cw or ccw

    for i in range(circlesteps):
        angle = ref_ang+(np.pi/4)*(-1)**dir
        delta=np.array([uwind*2,vwind*2]) + soar_rad * np.array([np.cos(angle),np.sin(angle)])
        #note tstep = approx 2 sec per pi/4 = 20 sec per circles/8 circle sectors - just a place holder for now
        ref_ang=angle
        new_pos=cur_pos + delta
        if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
           break
        directions.append(delta)
        trajectory.append(new_pos)
        circle_wt=0.5*np.exp(-1*float(i)/float(circlesteps)) #we assume thermalling eagle risk decays rapidly as it circles up
        step_wt=circle_wt
        track_weight.append(step_wt)
        cur_pos=new_pos
    
    # then glide in PAM dir approx 0.5 to 2 km
    
    #gliding distance/steps should be approx 0.5 to 2 km, and proportional to circling time, so
    glidesteps=int((500./step)+(circlesteps-32)/(160-32)*((2000./step)-(500./step)))
    
    #glide using the DRW function
    #new_pos = []
    for i in range(glidesteps):
        new_pos,step_wt = dir_random_walk(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,
                            maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,step=step,halfsector=5.)      
        glide_wt=0.5*float(i)/float(glidesteps) #note that weighting increases linearly from 0 to 0.5 as end of glide is reached
        step_wt=glide_wt
        delta = new_pos - trajectory[-1]
        if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
           break
        directions.append(delta)
        trajectory.append(new_pos)
        track_weight.append(step_wt)   
                         
    return new_pos,step_wt


#
# TODO for snapshot and seasonal modes, we are going to need wind functions like this 
# called from actions.py so that the WTK data can be accessed per each location in trajectory
# 
#    
#def get_wind_data():
#        wspeed, wdirn = self._get_interpolated_wind_conditions(
#            wtk_df[self.wtk_layers['wspeed']],
#            wtk_df[self.wtk_layers['wdirn']])
#    
#    return wspeed, wdirn  
#            
#def get_interpolated_wind_conditions(
#        self,
#        wspeed: np.ndarray,
#        wdirn: np.ndarray
#    ) -> Tuple[np.ndarray, np.ndarray]:
#        """ Interpolates wind speed and direction from wtk to terrain grid """
#        easterly = np.multiply(wspeed, np.sin(wdirn * np.pi / 180.))
#        northerly = np.multiply(wspeed, np.cos(wdirn * np.pi / 180.))
#        interp_easterly = self._interpolate_wtk_vardata(easterly)
#        interp_northerly = self._interpolate_wtk_vardata(northerly)
#        interp_wspeed = np.sqrt(np.square(interp_easterly) +
#                                np.square(interp_northerly))
#        interp_wdirn = np.arctan2(interp_easterly, interp_northerly)
#        interp_wdirn = np.mod(interp_wdirn + 2. * np.pi, 2. * np.pi)
#    
#    return interp_wspeed, interp_wdirn * 180. / np.pi
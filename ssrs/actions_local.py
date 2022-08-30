import numpy as np
from .actions import (searcharc_w,dir_random_walk)

def local_moves_mixedlift(trajectory,directions,track_weight,PAM,windspeed,winddir,threshold,lookaheaddist,
                          maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,
                          step=50.0,dist=50.0,halfsector=30,Nsearch=10,sigma=0.0,
                          rand_step_range=(5,10)):
                    
    """
    perform a sequence of moves based on updraft fields, then update move_dir and repeat
                    
    """
    cur_pos = trajectory[-1]
    last_move_dir = directions[-1]
 
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)

    move_dir=last_move_dir+np.random.uniform(-90,+90)
    if move_dir < 0:
        move_dir=360.+move_dir
    if move_dir > 360:
        move_dir=move_dir-360.
    #print('move dir =',move_dir)
    
    nsteps=np.random.randint(*rand_step_range) #do a sequence of steps for a particular move_dir
    for i in range(nsteps):
            
        wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                trajectory,directions,track_weight,move_dir,wo_interp,wt_interp,elev_interp,
                step=step,dist=dist,halfsector=halfsector,Nsearch=Nsearch)
    
                # take step based on search results, preferring movement upslope
                # if wt exceeds wo, do a thermal-gliding action
        if (wt_max1 > wo_max1) & (wt_max1 > threshold):
            delta = step * np.array([np.cos(best_dir_wt1),np.sin(best_dir_wt1)])
            new_pos = cur_pos + delta
            directions.append(move_dir)
            step_wt=0.5  #entering thermal
            trajectory.append(new_pos)
            track_weight.append(step_wt)
            new_pos,step_wt=thermal_soar_glide_local(trajectory,directions,track_weight,move_dir,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                    wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0)
            #delta = new_pos - trajectory[-1]
            #directions.append(move_dir)
            #trajectory.append(new_pos)
            #track_weight.append(step_wt)        
        elif wo_max1 > threshold and (elev_wo_max1 > elev_wo_max2):
            if sigma > 0:
                # add additional uncertainty
                best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(move_dir)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max2 > threshold and (elev_wo_max2 > elev_wo_max1):
            if sigma > 0:
            # add additional uncertainty
                best_dir_wo2 = np.random.normal(best_dir_wo2, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo2),np.sin(best_dir_wo2)])
            directions.append(move_dir)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max1 > threshold and (elev_wo_max1 - elev_cur_pos)>-20:
            if sigma > 0:
            # add additional uncertainty
                best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(move_dir)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max1 < threshold and wo_max1 >= 0.75*threshold and (elev_wo_max1 - elev_cur_pos)>0:
        #lift is near threshold so choose in that direction +-
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(30))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(move_dir)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)

        else:
            # no usable updraft found adjacent, do a directed random walk
            drwsteps=np.random.randint(*rand_step_range)
            for i in range(drwsteps):
                new_pos,step_wt = dir_random_walk(trajectory,directions,track_weight,move_dir,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                                      wo_interp,wo_sm_interp,wt_interp,elev_interp,
                                      step=step,halfsector=15.0)
                if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
                    break
                delta = new_pos - trajectory[-1]
                directions.append(move_dir)
                trajectory.append(new_pos)
                track_weight.append(step_wt)

        return new_pos,step_wt
    
    
    
def thermal_soar_glide_local(trajectory,directions,track_weight,move_dir,windspeed,winddir,threshold,lookaheaddist,maxx,maxy,
                             wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0,
                             wind_speed_pert_scalar=0.1, wind_dir_pert=10.0):
    """Three part movement is taken whenever thermal lift > threshold is encountered
            1. search for center of thermal using the searcharc() function
            2. circle and drift downwind for specified number of times (4 to 12 times)
            3. glide in the PAMward direction for a distance proportional to circling time             
    """                    
    cur_pos = trajectory[-1]
        
    #first, search outward arcs of 50 m out to 300 m to find center of thermal
    wt_best = np.zeros(6,dtype=float)
    loc_best = np.zeros(6,dtype=float)
    dir_best = np.zeros(6,dtype=float)
    rad_dist = np.zeros(6,dtype=float)
    for k,radius in enumerate(np.arange(50, 300.1, 50)):
        # search sector of wt in successive arcs to 300 m
        numsearchlocs = int(radius/5.)
        wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                        trajectory,directions,track_weight,move_dir,wo_sm_interp,wt_interp,elev_interp,
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
    
    directions.append(move_dir)
    step_wt=0.5  #base of thermal, assume moderate height
    trajectory.append(new_pos)
    track_weight.append(step_wt)
    
    cur_pos=new_pos
    
    #circle up and drift downwind in thermal
    
    #assume minimum of 4 loops (32 steps) and max of 20 loops (160 steps), proportional to wt of the thermal, with max of 8 m/s
    #note that it takes 8 steps for a full circle with angle = pi/4
    #circlesteps=int(32+((wt_globalmax-threshold)/(8-threshold))*(160-32))
    circlesteps=np.random.randint(32, 96) #number of steps circling
    soar_rad=np.random.uniform(20, 60) #soaring radius

    #TODO this assumes a constant windfield everywhere. Need to code for WTK with u(x,y) and v(x,y)
    
    #allow for some random variation in uniform wind speed and direction
    alpha=winddir
    np.random.seed() # is this needed?
    windspeed = windspeed * (1 + np.random.uniform(-wind_speed_pert_scalar, wind_speed_pert_scalar))
    winddir = winddir + np.random.uniform(-wind_dir_pert, wind_dir_pert)
    if winddir < 0:
        winddir=360.+winddir
    if winddir > 360:
        winddir=winddir-360.
        
    uwind=windspeed*np.cos(np.radians(270.-winddir))
    vwind=windspeed*np.sin(np.radians(270.-winddir))

    angle=move_dir

    dir=np.random.randint(1,3) #direction of soar, either cw or ccw

    for i in range(circlesteps):
        angle = angle+(np.pi/4)*(-1)**dir
        delta=np.array([uwind*2,vwind*2]) + soar_rad * np.array([np.cos(angle),np.sin(angle)])
        #note tstep = approx 2 sec per pi/4 = 20 sec per circles/8 circle sectors - just a place holder for now
        new_pos=cur_pos + delta
        if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
           print('break')
           break
        directions.append(move_dir)
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
        new_pos,step_wt = dir_random_walk(trajectory,directions,track_weight,move_dir,windspeed,winddir,threshold,lookaheaddist,
                            maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,step=step,halfsector=5.)      
        glide_wt=0.5*float(i)/float(glidesteps) #note that weighting increases linearly from 0 to 0.5 as end of glide is reached
        step_wt=glide_wt
        delta = new_pos - trajectory[-1]
        if not ((0.025*maxx < new_pos[0] < 0.975*maxx) and (0.025*maxy < new_pos[1] < 0.975*maxy)):
           print('break')
           break
        directions.append(move_dir)
        trajectory.append(new_pos)
        track_weight.append(step_wt)   
                         
    return new_pos,step_wt
    
    

def random_walk_local(trajectory,directions,track_weight,move_dir,windspeed,winddir,theshold,lookaheaddist,maxx,maxy,
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
    ref_ang = np.radians(move_dir)
                   
    ang0 = ref_ang - np.radians(halfsector)
    ang1 = ref_ang + np.radians(halfsector)
    
    rand_ang = ang0 + np.random.random()*(ang1 - ang0)
    delta = step * np.array([np.cos(rand_ang),np.sin(rand_ang)])
    step_wt=1. #typically a random movement might be to interact with another eagle or search for prey, etc
    
    return cur_pos + delta,step_wt

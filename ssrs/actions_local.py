import numpy as np

def local_moves(trajectory,directions,track_weight,PAM,windspeed,winddir,maxx,maxy,wo_interp,wo_sm_interp,wt_interp,elev_interp,
                    step=50.0,dist=50.0,look_ahead_dist=2000.0,halfsector=30,Nsearch=10,threshold=0.85,sigma=0.0):
                    
    """
        ***UNFINISHED CODE*** intent is to perform a sequence of moves based on lift and then update move_dir
                    
    """
    cur_pos = trajectory[-1]
    last_dir = directions[-1]  
    elev_cur_pos=elev_interp(cur_pos[0],cur_pos[1], grid=False)
    ref_ang = np.arctan2(last_dir[1],last_dir[0]) # previous movement angle (counterclockwise from +x)                       
    ref_ang_deg=np.degrees(ref_ang)
    ang0 = ref_ang_deg - 90
    ang1 = ref_ang_deg + 90
    
    move_dir = ang0 + np.random.random()*(ang1 - ang0)
    
    nsteps=np.random.randint(10, 20) #do a sequence of steps for a particular move_dir
    for i in range(nsteps):
            
        wo_max1,wt_max1,wo_max2,idx_womax1,idx_wtmax1,idx_womax2,best_dir_wo1,best_dir_wt1,best_dir_wo2,elev_wo_max1,elev_wo_max2 = searcharc_w(
                trajectory,directions,track_weight,move_dir,wo_interp,wt_interp,elev_interp,
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
            new_pos,step_wt=thermal_soar_glide(trajectory,directions,track_weight,move_dir,windspeed,winddir,maxx,maxy,
                    wo_interp,wo_sm_interp,wt_interp,elev_interp,step=50.0,halfsector=180.0)
            trajectory.append(new_pos)
            track_weight.append(step_wt)        
        elif wo_max1 > threshold and (elev_wo_max1 > elev_wo_max2):
            if sigma > 0:
                # add additional uncertainty
                best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(delta)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max2 > threshold and (elev_wo_max2 > elev_wo_max1):
            if sigma > 0:
            # add additional uncertainty
                best_dir_wo2 = np.random.normal(best_dir_wo2, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo2),np.sin(best_dir_wo2)])
            directions.append(delta)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max1 > threshold and (elev_wo_max1 - elev_cur_pos)>-20:
            if sigma > 0:
            # add additional uncertainty
                best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(sigma))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(delta)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)
        elif wo_max1 < threshold and wo_max1 >= 0.75*threshold and (elev_wo_max1 - elev_cur_pos)>0:
        #lift is near threshold so choose in that direction +-
            best_dir_wo1 = np.random.normal(best_dir_wo1, scale=np.radians(30))
            delta = step * np.array([np.cos(best_dir_wo1),np.sin(best_dir_wo1)])
            directions.append(delta)
            new_pos = cur_pos + delta
            step_wt=1.  #using orographic lift
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)

        else:
            # no usable updraft found adjacent, do a directed random walk step
            new_pos,step_wt = dir_random_walk(trajectory,directions,track_weight,move_dir,windspeed,winddir,maxx,maxy,
                                  wo_interp,wo_sm_interp,wt_interp,elev_interp,
                                  step=step,halfsector=15.0)
            if not ((0 < new_pos[0] < maxx) and (0 < new_pos[1] < maxy)):
                break
            trajectory.append(new_pos)
            track_weight.append(step_wt)

    return new_pos,step_wt

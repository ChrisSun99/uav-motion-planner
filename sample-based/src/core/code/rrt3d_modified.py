import numpy as np
import os, sys
# sys.path.append(os.getcwd())
from src.rrt.rrt import RRT
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot
import numpy as np
from copy import deepcopy


def add_points(path, num_points=3):
    """
    param:
        path:               list with tuples inside it
        num_points:         number of points you want to interpolate 
    return: 
        interpolated_path:  Same formation as path. 
    """
    
    interpolated_path = []
    ary_path = np.array(path)
    for i in range(len(path)-1):
        interpolated_path.append(path[i])
        delta = (ary_path[i+1] - ary_path[i])/(num_points+1)
        for j in range(num_points):
            temp = ary_path[i] + delta *(j+1)
            interpolated_path.append(tuple([temp[0], temp[1], temp[2]]))
    interpolated_path.append(path[-1])
    # interpolated_path = np.asarray(interpolated_path)
    # assert interpolated_path.shape[0] == path.shape[0] + (path.shape[0]-1) * num_points
    # assert interpolated_path.shape[1] == path.shape[1]
    assert len(interpolated_path) == ary_path.shape[0] + (ary_path.shape[0]-1) * num_points
    return interpolated_path 

def rescale_path(path, resolution):
    """
    Param: 
        path:       list with tuples inside it.
    return 
    """
    rescaled_path = []
    for i in range(len(path)):
        rescaled_path.append(tuple([path[i][0]*resolution, path[i][1]*resolution, path[i][2]*resolution]))
    
    return rescaled_path 

def rrt_search(world, start, goal, rrt_star=True, resolution=1, r=0.01, prc=0.1, max_samples=20000, margin=0.2):
    """
    Param:
        world:        World object representing the environment obstacles.
                      world_data, dict containing keys 'bounds' and 'blocks'
                        bounds, dict containing key 'extents'
                        extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                      blocks, list of dicts containing keys 'extents' and 'color'
                        extents, list of [xmin, xmax, ymin, ymax, zmin, zmax]
                        color, color specification
        r:            resolution of points to sample along edge when checking for collisions.
        prc:          probability of checking for a connection to goal
        max_samples:  max number of samples to take before timing out
    
    rerturn:
        path:   xyz position corrdinates. We get None if there is no way to get goal.
    """
    
    bound = deepcopy(world.world['bounds']['extents'])
    # Apply resolution
    X_dimensions = np.asarray(bound).reshape(3,2)
    X_dimensions_cp = resolution * deepcopy(X_dimensions)

    X_dimensions[:,0] = resolution * (X_dimensions[:,0] + margin)
    X_dimensions[:, 1] = resolution * (X_dimensions[:,1]- margin) 

    
    # obstacles
    blocks = deepcopy(world.world['blocks'])
    blocks = [block['extents'] for block in blocks]

    # Apply resolution
    Obstacles = np.asarray(blocks)[:, [0, 2, 4, 1, 3, 5]]
    Obstacles_cp = resolution * deepcopy(Obstacles)
    for item in Obstacles:
        item[0] = resolution*(item[0] - margin); item[1] = resolution*(item[1]-margin); item[2] = resolution*(item[2]-margin)
        item[3] = resolution*(item[3] + margin); item[4] = resolution*(item[4]+margin); item[5] = resolution*(item[5]+margin)

    # Obstacles = np.array(
    #     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
    #     (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])
    x_init = (resolution * start[0], resolution * start[1], resolution * start[2])#(0, 0, 0)  starting location.
    
    # Apply resolution
    if world.world['start'] is not None:
        x_init = (resolution*world.world['start'][0], resolution*world.world['start'][1], resolution*world.world['start'][2])
    
    x_goal = (resolution*goal[0], resolution*goal[1], resolution*goal[2]) #(100, 100, 100) goal location.
    if world.world['goal'] is not None:
        x_goal = (resolution*world.world['goal'][0], resolution*world.world['goal'][1], resolution*world.world['goal'][2])

    Q = np.array([(8, 4)])  # length of tree edges

    # create Search Space
    X = SearchSpace(X_dimensions, Obstacles)
    X_cp = SearchSpace(X_dimensions_cp, Obstacles_cp)

    # create rrt_search
    if rrt_star == False:
        rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
        path = rrt.rrt_search()
    else:
        rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count=32)
        path = rrt.rrt_star()

    # Add more intermediate points.

    plot = Plot("rrt_3d")
    plot.plot_tree(X_cp, rrt.trees)
    if path is not None:
        plot.plot_path(X_cp, path)
    plot.plot_obstacles(X_cp, Obstacles_cp)
    plot.plot_start(X_cp, x_init)
    plot.plot_goal(X_cp, x_goal)
    plot.draw(auto_open=False)

    path = rescale_path(path, 1./resolution)
    print(f"actual goal point={goal}")
    print(f"goal point = {path[-1]}")
    return np.asarray(path) 
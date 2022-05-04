from heapq import heappush, heappop  # Recommended.
import numpy as np
import copy

from flightsim.world import World

from occupancy_map import OccupancyMap  # Recommended.


def graph_search(world: World, resolution: tuple, margin: float,
                 start: np.ndarray, goal: np.ndarray, astar: bool) -> np.ndarray:
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # defined required matrices for execution of A* algorithm
    cost = np.full_like(occ_map.map, np.inf, dtype=float)
    pq = []
    parent_x = np.full(occ_map.map.shape, -1)
    parent_y = np.full(occ_map.map.shape, -1)
    parent_z = np.full(occ_map.map.shape, -1)
    visited = np.zeros_like(occ_map.map, dtype=bool)

    # initialize cost and pq for the start point
    cost[start_index] = 0.0
    heappush(pq, (cost[start_index], start_index))

    # main loop of A* algorithm
    while pq and (visited[goal_index] == False):

        # find the next unvisited node with minimum cost from pq
        expand_node = heappop(pq)
        expand_node_index = expand_node[1]
        while visited[expand_node_index] is True:
            expand_node = heappop(pq)
            expand_node_index = expand_node[1]
        # mark the node as visited
        visited[expand_node_index] = True

        # the node is connected to the nodes of +/- 1 coordinate difference in x,y,z direction from it
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # the coordinate of the neighbor node
                    neighbor_node_index = (expand_node_index[0] + i, expand_node_index[1] + j, expand_node_index[2] + k)
                    if not occ_map.is_occupied_index(neighbor_node_index):
                        if not visited[neighbor_node_index]:
                            neighbor_distance = cost[expand_node_index] + \
                                np.linalg.norm(np.array([i * resolution[0], j * resolution[1], k * resolution[2]]))
                            # update the cost of the neighbor node if the cost is reduced
                            if neighbor_distance < cost[neighbor_node_index]:
                                cost[neighbor_node_index] = neighbor_distance
                                # use a heuristic to calculate the expected cost if A* is used
                                if astar:
                                    neighbor_distance += \
                                        np.linalg.norm(np.array(goal) -
                                                       np.array(occ_map.index_to_metric_center(neighbor_node_index)))
                                # store the parent index
                                parent_x[neighbor_node_index] = expand_node_index[0]
                                parent_y[neighbor_node_index] = expand_node_index[1]
                                parent_z[neighbor_node_index] = expand_node_index[2]
                                # push the new cost to pq
                                heappush(pq, (neighbor_distance, neighbor_node_index))

    # return the start location if no path is found
    if visited[goal_index] == False:
        print("Path not found")
        return np.array(start).reshape((1, 3))
    # retrace the path to store it as an array
    path = [[goal[0], goal[1], goal[2]]]
    current_index = copy.deepcopy(goal_index)
    while parent_x[current_index] != -1:
        current_metric = occ_map.index_to_metric_center(current_index)
        path.append([current_metric[0], current_metric[1], current_metric[2]])
        current_index = (parent_x[current_index], parent_y[current_index], parent_z[current_index])
    path.append([start[0], start[1], start[2]])
    path = np.array(path)
    path = np.flip(path, axis=0)

    # Return the path
    return path

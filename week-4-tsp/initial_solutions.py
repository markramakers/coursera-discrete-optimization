from points import Point, length

from typing import List, Dict
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import time

M = 10000000000


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def get_kd_tree(points_list):
    X = np.array([[point.x, point.y] for point in points_list])
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    return kdt


@timeit
def create_initial_graph(points_list: List[Point], point_for_node: Dict[int, Point], k_connected_nodes: int = 2):
    kdt = get_kd_tree(points_list)

    graph = nx.Graph()
    for i, point in point_for_node.items():
        neighbours = kdt.query([point.to_list()], k_connected_nodes + 1)
        #print("edge: ", neighbours)
        import pdb
        # pdb.set_trace()
        edges = [
            (
                point,
                point_for_node[neighbour_index],
                {'weight': neighbours[0][0][i]}
            )
            for i, neighbour_index in enumerate(neighbours[1][0])
            if point != point_for_node[neighbour_index]]

        graph.add_edges_from(edges)
    # print("edges: ", graph.edges)
    # print("nodes: ", graph.nodes)
    return points_list, graph


@timeit
def create_mst(graph):
    return nx.minimum_spanning_tree(graph)


def test_create_initial_solution():
    expected = None
    result = create_initial_solution([Point(0, 1),
                                      Point(1, 0),
                                      Point(1, 1)])
    assert result == expected


def minimum_spanning_tree_prims(points_list):
    """
    https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
    :param points_list:
    :return:
    """

    mstSet = set()
    key_values = [0, M, M]

    while len(mstSet) < len(points_list):
        # Pick a vertex u which is not there in mstSet and has minimum key value.
        vertex_u = None

        # Include u to mstSet.
        mstSet.add(vertex_u)

        # Update key value of all adjacent vertices of u. To update the key values,
        # iterate through all adjacent vertices. For every adjacent vertex v,
        # if weight of edge u-v is less than the previous key value of v, update the key value as weight of u-v
@timeit
def christofides(points_list):
    
    G_ex = nx.Graph()
    G_ex.add_nodes_from(points_list)
    import itertools
    for p1, p2 in itertools.combinations(points_list, 2):
        G_ex.add_edge(p1, p2, weight=length(p1, p2))
    route = nx.approximation.christofides(G_ex, weight="weight")
    route = list(dict.fromkeys(route))

    tsp_initial = nx.Graph(list(nx.utils.pairwise(route)))
    return route, tsp_initial

@timeit
def create_distance_matrix(route):
    from scipy.spatial import distance_matrix as dm
    vectors = [[point.x, point.y] for point in route]
    distance_matrix = dm(vectors, vectors)
    return distance_matrix

@timeit
def two_opt(route, distance_matrix):
    from two_opt_solver import Solver
    # distance_matrix = [[length(pointA, pointB) for pointA in route]
                    #    for pointB in route]

    tsp = Solver(distance_matrix, range(0, len(route)))
    new_route, new_distance, distances = tsp.two_opt()
    new_points = [route[i] for i in new_route]
    return new_points

@timeit
def preorder_tree(mst):
    pre_ordered_tree = list(nx.dfs_preorder_nodes(mst))
    tsp_initial = nx.Graph(list(nx.utils.pairwise(pre_ordered_tree)))
    return pre_ordered_tree, tsp_initial

@timeit
def create_initial_solution(points_list):
    point_for_node = {
        i: point
        for i, point in enumerate(points_list)
    }
    location_for_point = {point: point.to_list() for point in points_list}
    # node_for_point = {
    #     point: i
    #     for i, point in enumerate(points_list)
    # }
    initial_solution, graph = create_initial_graph(points_list, point_for_node)
    # nx.draw(graph, pos=location_for_point)
    # plt.show(block=False)

    mst = create_mst(graph)
    # nx.draw(mst, pos=location_for_point)
    # plt.show(block=False)


    route, tsp_initial = preorder_tree(mst)

    if len(points_list) < 2000:
        route, tsp_initial = christofides(points_list)

    distance_matrix = create_distance_matrix(route)
    new_points = two_opt(route, distance_matrix)
    # print(new_points)



    # nx.draw(nx.Graph(list(nx.utils.pairwise(new_points))), pos=location_for_point)
    # plt.show(block=False)
    return new_points

if __name__ == "__main__":
    points_list = [Point(0, 1),
                   Point(1, 0),
                   Point(2, 2),
                   Point(2, 0),
                   Point(1, 1)]
    create_initial_solution(points_list)

  
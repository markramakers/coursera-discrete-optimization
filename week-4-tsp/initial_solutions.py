from points import Point

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
        print("edge: ", neighbours)
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
    print("edges: ", graph.edges)
    print("nodes: ", graph.nodes)
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


if __name__ == "__main__":
    points_list = [Point(0, 1),
                   Point(1, 0),
                   Point(2, 2),
                   Point(2, 0),
                   Point(1, 1)]
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
    nx.draw(graph, pos=location_for_point)
    plt.show()

    mst = create_mst(graph)
    nx.draw(mst, pos=location_for_point)
    plt.show()


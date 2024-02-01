#!/usr/bin/python
# -*- coding: utf-8 -*-


from points import Point, length
from initial_solutions import create_initial_solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]), i-1))

    # build a trivial solution
    
    # visit the nodes in the order they appear in the file
    #solution = range(0, nodeCount)

    # Create an inital solution based on an MST
    solution = create_initial_solution(points)

    # calculate the length of the tour
    obj = length(solution[-1], solution[0])
    for index in range(0, nodeCount - 1):
        obj += length(solution[index], solution[index + 1])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join([str(point.index) for point in solution])

    # print(nodeCount)
    # print(solution)
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

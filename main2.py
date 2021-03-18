import math

from igraph import *
from prettytable import PrettyTable
import numpy as np


def average_degree(G):
    return 2 * G.vcount() / G.ecount()


def average_g_distance(G):
    return np.mean(G.shortest_paths())


# n = 6
# p = 0.5
# rand_g = erdos_renyi_graph(n, p)
#
# print(rand_g.nodes)
# # [0, 1, 2, 3, 4, 5]
#
# print(rand_g.edges)
# # [(0, 1), (0, 2), (0, 4), (1, 2), (1, 5), (3, 4), (4, 5)]
#
# print(average_degree(rand_g))
#
# print(nx.average_shortest_path_length(rand_g))

# lattice_G = grid_graph(dim=(2, 3, 4))
# print(len(lattice_G))
#
# # lattice_G = grid_graph(dim=(range(7, 9), range(3, 6)))
# # print(len(lattice_G))
#
# print(average_degree(lattice_G))

def generate_random_g(s, p, n):
    av_degree = 0
    av_distance = 0
    for l in range(s):
        rand_g = Graph.Erdos_Renyi(n, p)
        av_degree += average_degree(rand_g)
        av_distance += average_g_distance(rand_g)
    av_degree /= s
    av_distance /= s
    return av_degree, av_distance


# for s in range(1, 10):
#     for p in numpy.arange(0.1, 1, 0.1):
#         for n in range(5, 10):
#             av_degree, av_distance = generate_random_g(s, p, n)
#             # print("Real Avg. Distance = " + str(av_distance))
#             # print(math.log(av_degree) if av_degree > 0 else "undefined")
#             # print("log N / log <k> = " + (str(math.log(n)/math.log(av_degree)) if av_degree > 0 else "undefined"))

# Constant <K>
def constant_k(s, p, n):
    t = PrettyTable(['S', 'p', 'N', '<K>', '<d>', 'log N', 'log <K>', 'log N / log <K>'])
    for x in range(3):
        av_degree, av_distance = generate_random_g(s, p, n)
        t.add_row([s, p, n, av_degree, av_distance, math.log(n),
                   math.log(av_degree) if av_degree > 0 else 'undefined',
                   math.log(n) / math.log(av_degree) if av_degree > 0 and math.log(
                       av_degree) != 0 else 'undefined'])

        p /= 10
        n *= 10
    print(t)


# Constant N
def constant_n(s, p, n):
    t = PrettyTable(['S', 'p', 'N', '<K>', '<d>', 'log N', 'log <K>', 'log N / log <K>'])
    for x in range(3):
        av_degree, av_distance = generate_random_g(s, p, n)
        t.add_row([s, p, n, av_degree, av_distance, math.log(n),
                   math.log(av_degree) if av_degree > 0 else 'undefined',
                   math.log(n) / math.log(av_degree) if av_degree > 0 and math.log(
                       av_degree) != 0 else 'undefined'])

        p /= 10
    print(t)


def lattice(n):
    t = PrettyTable(['D', 'N', '<k>', '<d>', 'N^(1/D)'])
    lattice_g1d = Graph.Lattice(dim=[n])
    t.add_row([1, len(lattice_g1d), average_degree(lattice_g1d), average_g_distance(lattice_g1d), len(lattice_g1d)])
    lattice_g2d = Graph.Lattice(dim=[n, n])
    t.add_row([2, len(lattice_g2d), average_degree(lattice_g2d), average_g_distance(lattice_g2d),
               len(lattice_g2d) ** (1 / 2)])
    lattice_g3d = Graph.Lattice(dim=[n, n, n])
    t.add_row([3, len(lattice_g3d), average_degree(lattice_g3d), average_g_distance(lattice_g3d),
               len(lattice_g3d) ** (1 / 3)])
    print(t)


constant_k(s=5, p=0.4, n=500)
# constant_n(s=1, p=0.8, n=2000)
# lattice(30)

import math

from networkx.generators.random_graphs import erdos_renyi_graph, fast_gnp_random_graph
from networkx import grid_graph, nx
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np


def average_degree(G):
    return 2 * len(G.edges) / len(G.nodes)


def average_g_distance(G):
    return nx.average_shortest_path_length(G) if nx.is_connected(G) else 0


def generate_random_g(s, p, n):
    av_degree = 0
    av_distance = 0
    for l in range(s):
        rand_g = fast_gnp_random_graph(n, p)
        av_degree += average_degree(rand_g)
        av_distance += average_g_distance(rand_g)
    av_degree /= s
    av_distance /= s
    return av_degree, av_distance


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
    lattice_g1d = grid_graph(dim=[n])
    t.add_row([1, n, average_degree(lattice_g1d), average_g_distance(lattice_g1d), n])

    n_g2d = n * n
    lattice_g2d = grid_graph(dim=[n, n])
    t.add_row([2, n_g2d, average_degree(lattice_g2d), average_g_distance(lattice_g2d),
               n_g2d ** (1 / 2)])

    n_g3d = n * n * n
    lattice_g3d = grid_graph(dim=[n, n, n])
    t.add_row([3, n_g3d, average_degree(lattice_g3d), average_g_distance(lattice_g3d),
               n_g3d ** (1 / 3)])
    print(t)


def nemoodar1(s, p, n):
    indices = np.arange(n) + 1

    random_g_distances = []
    random_g_eq_distances = []
    l1d_distances = []
    l1d_eq_distances = []
    first_value = 0
    last_none = 0
    for v in range(n):
        av_degree, av_distance = generate_random_g(s, p, v + 1)
        lattice_g1d = grid_graph(dim=[v + 1])
        av_dis = av_distance if av_distance > 0 else None
        if av_dis is None:
            last_none = v + 1
        else:
            first_value = v + 1 if first_value == 0 else first_value
        random_g_distances.append(av_dis)
        random_g_eq_distances.append(math.log(v + 1) / math.log(av_degree)
                                     if av_degree > 0 and math.log(av_degree) != 0
                                     else None)
        l1d_distances.append(average_g_distance(lattice_g1d))
        l1d_eq_distances.append(v + 1)
    print("First Connected Random Graph: " + str(first_value))
    print("Last Unconnected Random Graph: " + str(last_none))

    l2d_distances = [None] * n
    l2d_eq_distances = [None] * n
    n2 = int(n ** (1 / 2))
    for v2 in range(n2):
        lattice_g2d = grid_graph(dim=[v2 + 1, v2 + 1])
        n_g2d = (v2 + 1) * (v2 + 1)
        # print(n_g2d)
        l2d_distances[n_g2d - 1] = average_g_distance(lattice_g2d)
        l2d_eq_distances[n_g2d - 1] = n_g2d ** (1 / 2)

    l3d_distances = [None] * n
    l3d_eq_distances = [None] * n
    n3 = int(n ** (1 / 3))
    for v3 in range(n3):
        lattice_g3d = grid_graph(dim=[v3 + 1, v3 + 1, v3 + 1])
        n_g3d = (v3 + 1) * (v3 + 1) * (v3 + 1)
        # print(n_g3d)
        l3d_distances[n_g3d - 1] = average_g_distance(lattice_g3d)
        l3d_eq_distances[n_g3d - 1] = n_g3d ** (1 / 3)

    plt.plot(indices,
             random_g_distances, 'r-',
             # random_g_eq_distances, 'y-',
             l1d_distances, 'k-',
             # l1d_eq_distances, 'y-',
             l2d_distances, 'b+',
             # l2d_eq_distances, 'y+',
             l3d_distances, 'g+',
             # l3d_eq_distances, 'y+'
             )
    plt.axis(xmin=1, ymin=0)
    plt.xlabel('N')
    plt.ylabel('<d>')
    plt.savefig('nemoodar1.png')
    plt.show()


def nemoodar2(s, p, n):
    positions = np.arange(n) + 1
    # indices = []
    # for w in range(n):
    #     indices.append(math.log(w+1))

    random_g_distances = []
    l1d_distances = []
    first_value = 0
    last_none = 0
    for v in range(n):
        av_degree, av_distance = generate_random_g(s, p, v + 1)
        lattice_g1d = grid_graph(dim=[v + 1])
        av_dis = math.log(av_distance) if av_distance > 0 else None
        if av_dis is None:
            last_none = v + 1
        else:
            first_value = v + 1 if first_value == 0 else first_value
        random_g_distances.append(av_dis)
        av_l1d_dis = average_g_distance(lattice_g1d)
        l1d_distances.append(math.log(av_l1d_dis) if av_l1d_dis > 0 else None)
    print("First Connected Random Graph: " + str(first_value))
    print("Last Unconnected Random Graph: " + str(last_none))

    l2d_distances = [None] * n
    n2 = int(n ** (1 / 2))
    for v2 in range(n2):
        lattice_g2d = grid_graph(dim=[v2 + 1, v2 + 1])
        n_g2d = (v2 + 1) * (v2 + 1)
        av_l2d_dis = average_g_distance(lattice_g2d)
        l2d_distances[n_g2d - 1] = math.log(av_l2d_dis) if av_l2d_dis > 0 else None

    l3d_distances = [None] * n
    n3 = int(n ** (1 / 3))
    for v3 in range(n3):
        lattice_g3d = grid_graph(dim=[v3 + 1, v3 + 1, v3 + 1])
        n_g3d = (v3 + 1) * (v3 + 1) * (v3 + 1)
        av_l3d_dis = average_g_distance(lattice_g3d)
        l3d_distances[n_g3d - 1] = math.log(av_l3d_dis) if av_l3d_dis > 0 else None

    plt.plot(positions,
             random_g_distances, 'r-',
             l1d_distances, 'k-',
             l2d_distances, 'b+',
             l3d_distances, 'g+',
             )
    # plt.xticks(positions, indices)
    plt.axis(xmin=1, ymin=0)
    # plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('log N')
    plt.ylabel('log <d>')
    plt.savefig('nemoodar2.png')
    plt.show()


def calc_not_isolated(p):
    lnnn = 1
    n = 1
    while (p <= lnnn):
        n += 1
        lnnn = math.log(n) / n
    return n


# print(calc_not_isolated(0.8))
# constant_k(s=1, p=0.4, n=50)
# constant_n(s=1, p=0.8, n=2000)
# lattice(10)
nemoodar1(1, 0.2, 100)
# nemoodar2(10, 0.8, 100)

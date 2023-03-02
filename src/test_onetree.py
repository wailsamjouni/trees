import networkx as nx
import matplotlib.pyplot as plt
from GraphStructure import GraphStructure
from Graph import Graph
from Build import Build
from Overload import Overload
from Load import Load
import random
from collections import defaultdict

# generate_graph = Graph(11, 13, filename='graph1113.pkl') # my example
# generate_graph.save_to_file('graph1113.pkl')
# generate_graph = Graph(12, 9, filename='graph1211.pkl')  # test2
# generate_graph = Graph(17, 20, filename='graph1720.pkl')  # test3 gggg
# generate_graph = Graph(8, 8, filename='graph88.pkl') #test1
# generate_graph = Graph(8, 8, filename='graph1110.pkl')  # Important

# g_copy = GraphStructure(generate_graph.get_graph(),
#                         0, 5)

# network = Build(g_copy)
# network.build(0.4, "multitree")
# network.build(0.4, "edps")
# network.build(0.4, "onetree")


# load = Load(generate_graph.get_graph(), [0, 12, 2], 5)
# x = load.model_versions(0.4)
# load.load_all_versions(x)
# time_c = load.time_complexity(0.4)
# load.plot_complexity(time_c)

erdos = nx.erdos_renyi_graph(150, 1)

overload = Overload()
sources, destination = overload.sources_destination_random(erdos)
failed_edges = overload.generate_random_failed_edges(erdos,0.1)
print(sources, destination)

sizes = [10, 25,50,75,100,125,150, 175]

my_graph = erdos

edps = overload.compute_paths(my_graph,sources, destination,failed_edges, 'edps')
onetree = overload.compute_paths(my_graph,sources, destination,failed_edges, 'onetree')
multitree = overload.compute_paths(my_graph,sources, destination,failed_edges, 'multitree')
print(edps)
print(onetree)
print(multitree)

success_rate_edps, avg_path_length_edps = overload.avg_length_success_rate(edps)
success_rate_one, avg_path_length_edps = overload.avg_length_success_rate(onetree)
success_rate_multi, avg_path_length_edps = overload.avg_length_success_rate(multitree)

overload.plot_lengths_results(edps, 'Edps routes lengths')
overload.plot_lengths_results(onetree, 'One tree routes lengths')
overload.plot_lengths_results(multitree, 'Multiple Tree routes lengths')

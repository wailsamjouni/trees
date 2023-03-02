from GraphStructure import GraphStructure
from Build import Build
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import copy
import time
import timeit
import pandas as pd
import random


class Overload:
    def __init__(self):
        pass

    def path_to_edges(self, path):
        edges = []
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            edges.append(edge)
        return edges

    def get_just_paths(self, paths):
        # result = [path[2]
        #           for path in paths if path is not None and path[2] is not None]
        result = [path[2] for path in paths if path[1] is True]

        return result

    def resign_graph(self):

        g_copy = copy.deepcopy(self.graph)
        for structure in self.results:
            structure.graph.graph = g_copy
            
    def sources_destination_random(self, graph):
        
        dest_nodes = random.sample(list(graph.nodes()), 1)
        destination = dest_nodes[0]
        random_nodes = random.sample(list(graph.nodes()), 6)

        if destination in random_nodes:
            random_nodes.remove(destination)
            
        return random_nodes, destination
    
    def generate_random_failed_edges(self, graph, fraction):

        number_of_failed_edges = int(fraction * graph.number_of_edges())
        failed_edges = random.sample(graph.edges(), number_of_failed_edges)
        
        return failed_edges

    def compute_paths(self, erdos,sources, destination,failed_edges, version):

        paths = []
        for node in sources:

            g_copy = copy.deepcopy(erdos)
            graph_structure = GraphStructure(
                g_copy, node, destination)
            graph = Build(graph_structure)

            path = graph.build(failed_edges, version)
            paths.append(path)

        # computed_paths = self.get_just_paths(paths)
        # paths_edges = []

        # for path in computed_paths:
            # paths_edges.append(self.path_to_edges(path))

        # return paths_edges
        return paths

    def model_versions(self, fraction):

        edp_start = time.time()
        using_edps = self.compute_paths(fraction, version="edps")
        edp_end = time.time()
        edp = edp_end - edp_start

        one_start = time.time()
        using_one_tree = self.compute_paths(fraction, version="onetree")
        one_end = time.time()
        one = one_end - one_start

        multi_start = time.time()
        using_multiple_tree = self.compute_paths(fraction, version="multitree")
        multi_end = time.time()
        multi = multi_end - multi_start

        print(f'paths using edps: {using_edps} and time complexity is: {edp}')
        print(
            f'paths using one tree: {using_one_tree} and time complexity is: {one}')
        print(
            f'paths using multiple tree: {using_multiple_tree} and time complexity is: {multi}')

        return using_edps, using_one_tree, using_multiple_tree

    def load_versions(self, paths):

        if paths is not None and len(paths) > 0:
            all_edges = [frozenset(edge) for path in paths for edge in path]

            counter = Counter(all_edges)
            most_common = [tuple(edge) for edge, count in counter.most_common(
            ) if count == counter.most_common(1)[0][1]]

            edge_load = counter.most_common(1)[0][1]

            return most_common, edge_load

    def load_all_versions(self, paths):

        results = []
        i = 0

        for path_list in paths:
            result = self.load_versions(path_list)
            results.append(result)
            print(f'Result number {i+1} : {result}')
            i += 1

        return results

    def time_complexity(self, fraction):

        edp_start = time.time()
        using_edps = self.compute_paths(fraction, version="edps")
        edp_end = time.time()
        edp = edp_end - edp_start

        one_start = time.time()
        using_one_tree = self.compute_paths(fraction, version="onetree")
        one_end = time.time()
        one = one_end - one_start

        multi_start = time.time()
        using_multiple_tree = self.compute_paths(fraction, version="multitree")
        multi_end = time.time()
        multi = multi_end - multi_start

        return edp, one, multi

    def plot_complexity(self, complexities):

        edp, one, multi = self.time_complexity(complexities, fraction=0.4)

        plt.plot(edp, label="edp")
        plt.plot(one, label="one tree")
        plt.plot(multi, label="multi tree")

        plt.xlabel("Time complexity")
        plt.ylabel("Time(s)")
        plt.legend()
        plt.show()

    def model(self, sizes, fraction):

        edp_times = []
        one_times = []
        multi_times = []

        for number_nodes in sizes:
            random_graph = nx.gnp_random_graph(number_nodes, 0.3)
            # number_of_edges = random_graph.number_of_edges()
            # dest_node = random.sample(list(random_graph.nodes()), 1)
            # destination = dest_node[0]
            # random_nodes = random.sample(list(random_graph.nodes()), 4)

            # if destination in random_nodes:
            #     random_nodes.remove(destination)
                
            edp_times.append(timeit.timeit(
                lambda: self.compute_paths(random_graph, fraction, version="edps"), number=1) * 1000)

            one_times.append(timeit.timeit(
                lambda: self.compute_paths(random_graph, fraction, version="onetree"), number=1) * 1000)

            multi_times.append(timeit.timeit(
                lambda: self.compute_paths(random_graph, fraction, version="multitree"), number=1) * 1000)

        df = pd.DataFrame({
            "Sizes": sizes,
            # "Number of Edges": number_of_edges,
            "Edps": edp_times,
            "One Tree": one_times,
            "Multiple Trees": multi_times
        })

        plt.plot(df['Sizes'], df['Edps'], label='Edps')
        plt.plot(df['Sizes'], df['One Tree'], label='One Tree')
        plt.plot(df['Sizes'], df['Multiple Trees'], label='Multiple Trees')
        
        # plt.plot(df['Number of Edges'], df['Edps'], label='Edps')
        # plt.plot(df['Number of Edges'], df['One Tree'], label='One Tree')
        # plt.plot(df['Number of Edges'], df['Multiple Trees'], label='Multiple Trees')

        plt.xlabel('Number of nodes')
        plt.ylabel('Complexity in ms')
        plt.legend()

        plt.show()
        plt.savefig('Runtime5rounds.svg')
    def model_edps(self, sizes, fraction):

        edp_times = []

        for number_nodes in sizes:
            random_graph = nx.gnp_random_graph(number_nodes, 0.4)
            number_of_edges = random_graph.number_of_edges()
                
            edp_times.append(timeit.timeit(
                lambda: self.compute_paths(random_graph, fraction, version="edps"), number=1))

        df = pd.DataFrame({
            "Sizes": sizes,
            "Edps": edp_times,
        })

        plt.plot(df['Sizes'], df['Edps'], label='Edps')

        plt.xlabel('Number of nodes')
        plt.ylabel('Complexity')
        plt.legend()

        plt.show()
        plt.savefig('Runtime.svg')
    def model_one(self, sizes, fraction):

        onetree_times = []

        for number_nodes in sizes:
            random_graph = nx.gnp_random_graph(number_nodes, 0.4)
            number_of_edges = random_graph.number_of_edges()
                
            onetree_times.append(timeit.timeit(
                lambda: self.compute_paths(random_graph, fraction, version="onetree"), number=1))

        df = pd.DataFrame({
            "Sizes": sizes,
            "One Tree": onetree_times,
        })

        plt.plot(df['Sizes'], df['One Tree'], label='Edps')

        plt.xlabel('Number of nodes')
        plt.ylabel('Complexity')
        plt.legend()

        plt.show()
        plt.savefig('Runtime.svg')

    def avg_length_success_rate(self, paths):
    
        total_count = 0
        succ_count = 0
        path_length = 0
    
        for path in paths:
        
            if path[1] is True:
                succ_count += 1
                if path[2] is not None:
                    if len(path[2]) > 0:
                        path_length += len(path[2])
            total_count +=1
        
        avg_path_length = path_length / succ_count if succ_count > 0 else 0
        success_rate = succ_count / total_count * 100
    
        return success_rate, avg_path_length

# edp_paths = overload.compute_paths(random_graph, 0.3, 'edps')
# one_tree_paths = overload.compute_paths(random_graph, 0.3, 'onetree')
# multi_tree_paths = overload.compute_paths(random_graph, 0.3, 'multitree')

# success_rate_edps, avg_path_length_edps = avg_length_success_rate(edp_paths)
# success_rate_one, avg_path_length_edps = avg_length_success_rate(one_tree_paths)
# success_rate_multi, avg_path_length_edps = avg_length_success_rate(multi_tree_paths)

    def plot_lengths_results(self, paths, title):
    
        paths_successed = [path for path in paths if path[1] and path[2]]
        lengths = [len(path[2]) for path in paths_successed]
    
        plt.hist(lengths, bins=10, density=True, edgecolor='black', rwidth=0.8)
        plt.xlabel('Route length')
        plt.ylabel('Frequency (%)')
        plt.title(title)
        plt.show()
    
# plot_lengths_results(edp_paths, 'Edps routes lengths')
# plot_lengths_results(one_tree_paths, 'One tree routes lengths')
# plot_lengths_results(multi_tree_paths, 'Multiple Tree routes lengths')

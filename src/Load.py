from GraphStructure import GraphStructure
from Build import Build
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import copy
import time


class Load:
    def __init__(self, graph, sources, destination):

        try:
            self.graph = graph
            self.sources = sources
            self.destination = destination
            self.results = []

            for source in self.sources:
                try:
                    if source not in self.graph.nodes():
                        raise ValueError(
                            f"Source node {source} not found in graph")
                    if self.destination not in self.graph.nodes():
                        raise ValueError(
                            f"Destination node {destination} not found in graph")

                    g_copy = copy.deepcopy(self.graph)
                    graph_structure = GraphStructure(
                        g_copy, source, destination)

                    result = Build(graph_structure)
                    self.results.append(result)

                except Exception as error:
                    print(
                        f"Error creating GraphStructure for source {source}: {error}")

        except Exception as error:
            print(f'Error creating : {error}')

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

    def compute_paths(self, fraction, version):

        paths = []
        for result in self.results:
            result.graph.graph = copy.deepcopy(self.graph)
            path = result.build(fraction, version)
            paths.append(path)

        computed_paths = self.get_just_paths(paths)
        paths_edges = []

        for path in computed_paths:
            paths_edges.append(self.path_to_edges(path))

        return paths_edges

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

import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle


class Graph:
    def __init__(self, number_nodes, number_edges, filename=None):

        self.number_nodes = number_nodes
        self.number_edges = number_edges
        self.nodes = range(self.number_nodes)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)

        if filename:
            self.load_graph_from_file(filename)
        else:
            self.add_edges()

    def add_edges(self):
        while nx.is_connected(self.graph) == False:
            self.graph.clear()
            self.graph.add_nodes_from(self.nodes)
            for i in range(self.number_edges):
                u, v = random.sample(self.nodes, 2)
                self.graph.add_edge(u, v)

    def save_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            self.graph = pickle.load(f)

    def draw(self):
        # pos = nx.spring_layout(self.graph)
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, with_labels=True, pos=pos)
        plt.show()

    def get_graph(self):
        return self.graph

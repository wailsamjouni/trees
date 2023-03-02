from GraphStructure import GraphStructure
import networkx as nx
import logging
import matplotlib.pyplot as plt


class Build:

    def __init__(self, graph: GraphStructure):
        self.graph = graph

    def setup(self):

        self.graph.set_attributes_to_zero()
        edge_disjoint_paths = self.graph.compute_and_sort_edps()
        edge_attrs = nx.get_edge_attributes(self.graph.graph, "attr")
        self.graph.number_the_computed_edps(edge_disjoint_paths)
        destination_incidents = self.graph.find_destination_incidents()
        edge_attrs_after = nx.get_edge_attributes(self.graph.graph, "attr")
        # failed_edges_random = self.graph.generate_random_failed_edges(fraction)
        # failed_edges_random = [(0, 13), (2, 11)]
        # failed_edges_random = [(14, 7)]
        # shortest_path_after_removing_failededges = self.graph.compute_shortest_path(
        #     failed_edges_random)
        # return edge_disjoint_paths, edge_attrs, destination_incidents, edge_attrs_after, failed_edges_random, shortest_path_after_removing_failededges
        return edge_disjoint_paths, edge_attrs, destination_incidents, edge_attrs_after

    def sort_and_get_attr(self):
        
        try:
            edps = sorted(list(nx.edge_disjoint_paths(
            self.graph.graph, self.graph.source, self.graph.destination)), key=len, reverse=False)

            edge_attrs_after = nx.get_edge_attributes(self.graph.graph, "attr")

            return edps
        
        except nx.NetworkXNoPath:
            return []

    def build_tree_attr(self, number_attr):

        Tree = nx.Graph()
        for first_node, second_node, data in self.graph.graph.edges(data=True):
            if data["attr"] == str(number_attr):
                Tree.add_edge(first_node, second_node)
        return Tree

    def draw_graph(self, Tree):
        pos = nx.circular_layout(Tree)
        nx.draw(Tree, with_labels=True, pos=pos)
        plt.show()

    def edps_method(self, edge_disjoint_paths, failed_edges_random):

        self.graph.remove_failed_edges(failed_edges_random)

        # logger.debug(f'Routing with edge disjoint paths choosed')

        # logger.debug(f'The sorted EDPs are : {edge_disjoint_paths}')

        # logger.info('Routing has just begun')
        edp_number, destination_reached, path_to_destination = self.graph.routing_with_edps(
            edge_disjoint_paths, failed_edges_random)

        # if destination_reached:
            # logger.info(
                # f'Destination node "{self.graph.destination}" can be reached through this path {path_to_destination} number {edp_number}')
        # else:
            # logger.critical(
                # f'Destination node "{self.graph.destination}" can not be reached')
        return [edp_number, destination_reached, path_to_destination]

    def one_tree_method(self, edge_disjoint_paths, destination_incidents, failed_edges_random):

        number_attr = self.graph.oneTree(edge_disjoint_paths, reverse=True)
        # logger.debug(f'One tree choosed')

        # edps, edge_attrs_after = self.sort_and_get_attr()
        edps = self.sort_and_get_attr()

        # logger.debug(f'The sorted EDPs are : {edps}')

        # logger.info(
            # f'Updated edps between {self.graph.source} and {self.graph.destination} are : {edps}')

        # logger.error(
            # f'The edge attributes after extending the edp {edps[len(edps) - 1]} are : {edge_attrs_after}')

        self.graph.disconnect_the_edges_of_the_destination()

        self.graph.prune_akt(destination_incidents, number_attr)

        self.graph.remove_failed_edges(failed_edges_random)

        Tree = self.build_tree_attr(number_attr)
        # self.draw_graph(Tree)

        edge_attrs_pruned = nx.get_edge_attributes(
            self.graph.graph, "attr")
        # logger.debug(
            # f'Attributes after pruning the tree : {edge_attrs_pruned}')

        # logger.warning(f'Tree is : {Tree}')

        # logger.info('Routing has just begun')
        tree_attr, destination_reached, path = self.graph.routing_with_one_tree(edps, Tree,
                                                                                destination_incidents, failed_edges_random)
        # logger.debug(f'Tree with attr "{len(edps)}" is : {Tree.edges}')

        # if destination_reached:
            # logger.info(
                # f'Destination node "{self.graph.destination}" can be reached through this path {path} number {tree_attr} using one tree')
        # else:
            # logger.critical(
                # f'Destination node "{self.graph.destination}" can not be reached through the tree number {tree_attr} using one tree')
        # self.draw_graph(self.graph.graph)
        return [tree_attr, destination_reached, path]

    def multiple_tree_method(self, edge_disjoint_paths, destination_incidents, failed_edges_random):
        # print(f'Failed edges in multiple tree are : {failed_edges_random}')

        # logger.debug(f'Multiple tree choosed')

        # edps, edge_attrs_after = self.sort_and_get_attr()
        edps = self.sort_and_get_attr()

        # logger.debug(f'The sorted EDPs are : {edps}')

        if edge_disjoint_paths is not None and  len(edge_disjoint_paths) > 0:
            
            for edge_disjoint_path in edge_disjoint_paths:

                number_attr = self.graph.tree_based_edge(
                edge_disjoint_paths, edge_disjoint_path)
                # logger.critical(f'The tree choosed is : {number_attr}')

                self.graph.disconnect_the_edges_of_the_destination()

                self.graph.prune_akt(destination_incidents, number_attr)

        self.graph.remove_failed_edges(failed_edges_random)

        # logger.info('Routing has just begun')
        tree_attr, destination_reached, path = self.graph.routing_with_multiple_tree(edps,
                                                                                     destination_incidents, failed_edges_random)
        # print(f'{tree_attr}, {destination_reached}, {path}')

        # if destination_reached:
            # logger.info(
                # f'Destination node "{self.graph.destination}" can be reached through this path {path} number {tree_attr} using multiple tree')
        # else:
            # logger.critical(
                # f'Destination node "{self.graph.destination}" can not be reached through {path} number {tree_attr} using multiple tree')
        return [tree_attr, destination_reached, path]

    def build(self, failed_edges_random, version=None):

        # logger = self.graph._get_logger('build.log')

        edge_disjoint_paths, edge_attrs, destination_incidents, edge_attrs_after = self.setup()
        # print(f'edpassssssssssssssssssssssssssssssssss: {len(edge_disjoint_paths)}')

        # logger.info(
            # f'EDPS between {self.graph.source} and {self.graph.destination} are :{edge_disjoint_paths}')
        # logger.info(f'Attributes before extending EDP : {edge_attrs}')
        # logger.info(f'Destination neighbors are : {destination_incidents}')
        # logger.info(
            # f'Attributes after numbering the EDPs : {edge_attrs_after}')
        # logger.warning(f'Failed edges are : {failed_edges_random}')
        # logger.info(
        #     f'Shortest path after removing failed edges: {shortest_path_after_removing_failededges}')

        if version == "edps":

            return self.edps_method( edge_disjoint_paths, failed_edges_random)


        if version == "onetree":

            return self.one_tree_method( edge_disjoint_paths,
                                        destination_incidents, failed_edges_random)

        elif version == "multitree":

            return self.multiple_tree_method( edge_disjoint_paths, destination_incidents, failed_edges_random)

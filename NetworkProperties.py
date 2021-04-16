# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:55:10 2021

@author: abelj
"""

import numpy as np
import networkx as nx


class NetworkProperties():
    """Class representation of Network Properties."""

    def __init__(self, G):
        """
        Initialize a new instance of Network Properties.

        Parameters
        ----------
        G : networkx graph
            Networkx Graph to get property.
        """
        self.__N = G.order()
        self.__L = G.size()
        self.__avg_degree = float(self.__L) / self.__N
        self.__density = self.__avg_degree / (self.__N - 1)

        # Check degree
        degrees = [k for node, k in nx.degree(G)]
        self.__max_degree = np.max(degrees)
        self.__min_degree = np.min(degrees)

        self.__is_connected = nx.is_connected(G)

        # Calculate clustering coefficient
        cc = nx.clustering(G)
        self.__avg_clustering_coeff = sum(cc.values()) / len(cc)

        # Pair-wise distance matrix
        self.__distance = dict(nx.all_pairs_shortest_path_length(G))
        self.__node_list = G.nodes()

        self.__D = np.zeros((len(self.__node_list), len(self.__node_list)))
        for i, node_i in enumerate(self.__node_list):
            for j, node_j in enumerate(self.__node_list):
                if node_j in self.__distance[node_i].keys():
                    self.__D[i, j] = self.__distance[node_i][node_j]

        # Calculate average path length
        if self.__is_connected:
            self.__avg_shortest_path = nx.average_shortest_path_length(G)
        else:
            self.__avg_shortest_path = np.mean(
                [path for node_i, edges in self.__distance.items()
                 for node_j, path in edges.items()]
            )

        # Find the max cliques
        self.__cliques = list(nx.find_cliques(G))
        
        self.__G = G
        pass

    @property
    def N(self):
        """Return total number of nodes of the graph."""
        return self.__N

    @property
    def L(self):
        """Return total number of edges of the graph."""
        return self.__L

    @property
    def avg_degree(self):
        """Return the average degree of the graph."""
        return self.__avg_degree

    @property
    def density(self):
        """Return density of the graph."""
        return self.__density

    @property
    def max_degree(self):
        """Return maximum number of degree."""
        return self.__max_degree

    @property
    def min_degree(self):
        """Return minimum number of degree."""
        return self.__min_degree

    @property
    def avg_shortest_path(self):
        """Return average shortest path length."""
        return self.__avg_shortest_path

    @property
    def avg_clustering_coeff(self):
        """Return average clustering coefficient."""
        return self.__avg_clustering_coeff

    @property
    def pairwise_distance_matrix(self):
        """Return pairwise distance matrix."""
        return self.__D

    @property
    def shortest_path_length_max(self):
        """Return shortest path length max."""
        return np.max(self.__D)

    @property
    def shortest_path_length_max_edge(self):
        """Return edges shortest path length max."""
        idx = np.argmax(self.__D)
        node_list = list(self.__node_list)
        i = idx // len(node_list)
        j = idx % len(node_list)

        return (node_list[i], node_list[j])

    @property
    def is_connected(self):
        """Return True if graph is connected otherwise False."""
        return self.__is_connected

    @property
    def maximal_cliques(self):
        """Return the maximum length of cliques found."""
        cliques = self.__cliques
        if len(cliques) > 0:
            cliques = sorted(cliques, key=len, reverse=True)
            max_cliques = len(cliques[0])
        else:
            max_cliques = 0

        return max_cliques

    def get_maximal_cliques_subgraphs(self):
        """
        Get list subgraphs of the clique with the max length.

        Returns
        -------
        max_cliques_subgraphs : list
            List of subgraphs of cliques with the max length.

        """
        return [self.__G.subgraph(set(nodes))
                for nodes in self.__cliques
                if len(nodes) == self.maximal_cliques]

    def __str__(self):
        """Return string of statistics."""
        N = f'N = {self.N}'
        L = f'L = {self.L}'
        avg_degree = f'avg_degree = {self.avg_degree}'
        max_degree = f'max_degree = {self.max_degree}'
        min_degree = f'min_degree = {self.min_degree}'
        density = f'density = {self.density}'
        avg_shortest_path = f'avg_shortest_path = {self.avg_shortest_path}'
        avg_clustering_coeff = ('avg_clustering_coeff = {}'
                                .format(self.avg_clustering_coeff))
        shortest_path_length_max = ('shortest_path_length_max = {}'
                                    .format(self.shortest_path_length_max))
        shortest_path_length_max_edge = (
            'shortest_path_length_max_edge = {}'
            .format(self.shortest_path_length_max_edge))
        is_connected = ('is_connected = {}'
                        .format(self.is_connected))
        maximal_cliques = ('maximal_cliques = {}'.format(self.maximal_cliques))
        string = ('Network Properties:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}'
                  '\n{}\n{}\n{}\n{}'
                  .format(N, L, avg_degree, max_degree, min_degree, density,
                          avg_shortest_path, avg_clustering_coeff,
                          shortest_path_length_max,
                          shortest_path_length_max_edge,
                          is_connected, maximal_cliques
                          ))
        return string

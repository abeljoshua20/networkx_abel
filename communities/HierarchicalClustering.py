# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:29:28 2021

@author: abelj
"""
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
from communities.NetworkCommunities import NetworkCommunities


class HierarchicalClustering(NetworkCommunities):
    """Class representation of Hierarchical Clustering for communities."""

    def __init__(self, t, method='average'):
        """
        Initialize a new instance of HierarchicalClustering.

        Parameters
        ----------
        t : int
            threshold to look of the distance between communities.
        method : {'average', 'single', 'ward', 'complete', 'centroid'}
            method for hierarchical clustering

        Returns
        -------
        None.

        """
        NetworkCommunities.__init__(self)
        self.__t = t
        if method not in {'average', 'single', 'ward', 'complete', 'centroid'}:
            raise Exception(f'Invalid method {method}')
        self.__method = method
        self.__Z = None

    @property
    def t(self):
        """Return threshold of clustering."""
        return self.__t

    @property
    def method(self):
        """Return method used for clustering."""
        return self.__method

    @property
    def Z(self):
        """Return hierarchical clustering encoded as linkage matrix."""
        return self.__Z

    def fit(self, G):
        """
        Fit the graph to find communities using Hierarchical Clustering.

        Parameters
        ----------
        G : networkx graph
            Graph to find communities.

        Returns
        -------
        None.

        """
        self._G = G.copy()
        Z, membership = self.__create_hc(self._G)
        # List of set of nodes in graph G
        components = {}
        for node, memb in zip(set(G.nodes()), membership):
            if memb not in components.keys():
                components[memb] = {node}
            else:
                components[memb].add(node)

        self._community_count = len(components)
        self._components = components.values()
        self.__Z = Z

    def __create_hc(self, G):
        """
        Perform Hierarchical Clustering to path lengths of graph G.

        Parameters
        ----------
        G : networkx graph
            Graph to find communities.

        Returns
        -------
        Z, membership : 2 tuple of ndarray and array-like
            Z is the hierarchical clustering encoded as linkage matrix.
            membership is the community of each node.

        """
        labels = G.nodes()
        path_length = nx.all_pairs_shortest_path_length(G)

        # Setup the distance matrix D
        distances = np.zeros((len(G), len(G)))
        # For non-numbered nodes, get index of each node
        distance_indx = dict(zip(set(labels), range(len(G))))
        # Fill the distance matrix
        for node, info in path_length:
            for other_node, length in info.items():
                # Get the index of each node in the matrix
                i = distance_indx[node]
                j = distance_indx[other_node]
                # If the same node
                if node == other_node:
                    distances[i][j] = 0
                else:
                    distances[i][j] = length
                    distances[j][i] = length

        Z = hierarchy.linkage(distances, method=self.__method,
                              optimal_ordering=True)
        membership = hierarchy.fcluster(Z, t=self.__t,
                                        criterion='distance')

        return Z, membership

    def plot_dendogram(self, ax):
        """
        Plot dendogram of trained hierarchical cluster.

        Parameters
        ----------
        ax : AxesSubplot:
            Place plot centrality on ax.

        Returns
        -------
        ax : AxesSubplot
            axes subplot of the plot.

        """
        hierarchy.dendrogram(self.__Z, ax=ax)
        return ax

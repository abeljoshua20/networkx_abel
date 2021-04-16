# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:13:19 2021

@author: abelj
"""

import networkx as nx
from operator import itemgetter


import os
import sys
import inspect

current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import networkx_abel as nxa
from networkx_abel.communities.NetworkCommunities import NetworkCommunities


class GirvanNewman(NetworkCommunities):
    """Class representation of Girvan Newman Algorithm."""

    def __init__(self, min_communities):
        """
        Initializes a new instance of GirvanNewman.

        Parameters
        ----------
        min_communities : int
            Number of minimum communities.

        Returns
        -------
        None.

        """
        NetworkCommunities.__init__(self)
        self.__min_communities = min_communities
        self.__fitted_graph = None

    @property
    def fitted_graph(self):
        """Return fitted graph after removing edges."""
        return self.__fitted_graph

    def fit(self, G):
        """
        Fit the graph to find communities using Girvan Newman algo.

        Parameters
        ----------
        G : networkx graph
            Graph to find communities.

        Returns
        -------
        None.

        """
        self._G = G.copy()

        if G.order() < self.__min_communities:
            self.__components = set(G.nodes())
            self._community_count = len(nx.connected_components(G))
            return

        new_G = self._G.copy()
        new_components = nxa.connected_component_subgraphs(new_G)
        while len(new_components) < self.__min_communities:
            new_G, new_components = self.__girvan_newman(self._G)

        # Count the number of communities found
        self.__fitted_graph = new_G
        self._community_count = len(new_components)
        self._components = new_components

    def __find_best_edge(self, G):
        """
        Find the best edge of the graph using highest betweenness centrality.

        Returns
        -------
        best_edge : 2 tuple
            edge with highest betweenness centrality.

        """
        betweenness = nx.edge_betweenness_centrality(G)
        # Edge as key, value as betweenness
        betweenness_items = betweenness.items()
        # Sort based on betweenness desc
        return sorted(betweenness_items, key=itemgetter(1), reverse=True)[0][0]

    def __girvan_newman(self, G):
        """
        Find the communities of each node in self.__G using girvan newman algo.

        Parameters
        ----------
        G : networkx graph
            Graph to find communities.

        Returns
        -------
        graph, components : 2 tuple of networkx graph and list of sets
            Graph with more than one components.
            List of set of nodes per component.

        """
        components = nxa.connected_component_subgraphs(G)
        while len(components) == 1:
            G.remove_edge(*self.__find_best_edge(G))
            components = nxa.connected_component_subgraphs(G)

        new_components = [set(c.nodes()) for c in components]
        return G, new_components

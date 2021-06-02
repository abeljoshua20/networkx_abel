# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:52:05 2021

@author: abelj
"""
import community
import networkx as nx
from .NetworkCommunities import NetworkCommunities


class LouvainMethod(NetworkCommunities):
    """Class representation of the Louvan Method to find communities."""

    def __init__(self):
        """
        Initialize a new instance of the LouvainMethod.

        Returns
        -------
        None.

        """
        NetworkCommunities.__init__(self)

    def fit(self, G):
        """
        Fit the graph to find communities using Louvain Method.

        Parameters
        ----------
        G : networkx graph
            Graph to find communities.

        Returns
        -------
        None.

        """
        self._G = G.copy()
        partition = community.best_partition(self._G)
        self._community_count = len(set(partition.values()))
        components = dict()
        for node, memb in partition.items():
            if memb not in components.keys():
                components[memb] = {node}
            else:
                components[memb].add(node)
        self._components = components.values()

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:04:41 2021

@author: abelj
"""


class NetworkCommunities():
    """Class representation of network communities."""

    def __init__(self):
        """
        Initialize a new instance of NetworkCommunities.

        Returns
        -------
        None.

        """
        self._node_membership = None
        self._community_count = None
        self._components = None
        self._G = None
        pass

    @property
    def node_membership(self) -> dict:
        """Return dictionary where node as key value as members."""
        return self._node_membership

    @property
    def community_count(self) -> int:
        """Return number of communities in the network."""
        return self._community_count

    @property
    def components(self):
        """Return the list of set of nodes per community for the graph."""
        return self._components

    def fit(self, G):
        """Pass the network communities model fit."""
        pass

    def transform(self):
        """
        Return a list of membership of each nodes in the graph.

        Returns
        -------
        result : array like
            List of membership of each nodes.

        """
        self._node_membership = dict()
        for membership, nodes in enumerate(self._components):
            for node in nodes:
                self._node_membership[node] = membership

        return [self._node_membership[node] for node in self._G.nodes()]

    def fit_transform(self, G):
        """Pass the network communities model fit_transform."""
        self.fit(G)
        return self.transform()

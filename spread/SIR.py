# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:20:19 2021

@author: abelj
"""

import networkx as nx
import numpy as np
from .SpreadingPhenomena import SpreadingPhenomena


class SIR(SpreadingPhenomena):
    """Class representation of SIR spreading phenomena."""

    def __init__(self, G, infected_nodes, beta, mu):
        """
        Initialize a new instance of SIR spreading phenomena.

        Parameters
        ----------
        G : networkx graph
            Graph to simulate epidemic.
        infected_nodes : list
            List of infected nodes.
        beta : float
            likelihood that the susceptible will be infected.
        mu : float
            likelihood that the an individual will recover from infection.

        Returns
        -------
        None.

        """
        SpreadingPhenomena.__init__(self, G)
        self.__infected_nodes = infected_nodes
        self.__beta = beta
        self.__mu = mu
        node_status = dict(map(lambda x: (x, 1) if x in self.__infected_nodes
                               else (x, 0), G.nodes()))
        nx.set_node_attributes(self._G, node_status, name='status')

    @property
    def infected_nodes(self):
        """Get list of infected nodes."""
        return list(filter(lambda x: x[1]['status'] == 1,
                           self._G.nodes(data=True)))

    def iterate(self, n):
        """
        Iterate n number of spreading phenomena using SIR model.

        Parameters
        ----------
        n : int
            number of iterations in the spreading phenomena.

        Returns
        -------
        graph : networkx graph
            yield graph of spreaded phenomena for each iteration

        """
        for i in range(n):
            self._G = self.__epidemic_spread(self._G)
            yield self._G

    def __epidemic_spread(self, G):
        """
        Simulate the spread of epidemic for the next iteration.

        Parameters
        ----------
        G : networkx graph
            graph to simulate the spread.

        Returns
        -------
        G : networkx graph
            simulated spread of the graph, changed weights.

        """
        infecteds = list(filter(lambda x: x[1]['status'] == 1,
                                G.nodes(data=True)))
        for node, attr in infecteds:
            for neighbor_nodes in G.neighbors(node):
                if G.nodes[neighbor_nodes]['status'] == 0:
                    # If the neighbor is Susceptible
                    # it has a beta probability of getting infected
                    new_status = np.random.choice([0, 1], p=[1-self.__beta,
                                                             self.__beta])
                    G.nodes[neighbor_nodes]['status'] = new_status
            if G.nodes[node]['status'] == 1:
                # If infected, check the probability of recovery
                new_status = np.random.choice([1, 2], p=[1-self.__mu,
                                                         self.__mu])
                G.nodes[node]['status'] = new_status
        return G

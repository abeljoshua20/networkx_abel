# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:38:17 2021

@author: abelj
"""

import networkx as nx
import numpy as np
from .SpreadingPhenomena import SpreadingPhenomena


class SEIR(SpreadingPhenomena):
    """Class representation of SEIR spreading phenomena."""

    def __init__(self, G, infected_nodes, beta, sigma, gamma):
        """
        Initialize a new instance of SEIR spreading phenomena.

        Parameters
        ----------
        G : networkx graph
            Graph to simulate epidemic.
        infected_nodes : list
            List of infected nodes.
        beta : float
            likelihood that the susceptible will be exposed.
        sigma : float
            likelihood that the exposed individual will be infected..
        gamma : float
            likelihood that the an individual will recover from infection.

        Returns
        -------
        None.

        """
        SpreadingPhenomena.__init__(self, G)
        self.__infected_nodes = infected_nodes
        self.__beta = beta
        self.__gamma = gamma
        self.__sigma = sigma
        node_status = dict(map(lambda x: (x, 2) if x in self.__infected_nodes
                               else (x, 0), G.nodes()))
        nx.set_node_attributes(self._G, node_status, name='status')

    @property
    def infected_nodes(self):
        """Get list of infected nodes."""
        return list(filter(lambda x: x[1]['status'] == 2,
                           self._G.nodes(data=True)))

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

    def __init__(self, G, infected_nodes, beta, mu, social_distancing=0,
                 vaccinated_nodes=[], vaccine_efficacy=0, seed=None):
        """
        Initialize a new instance of SIR spreading phenomena.

        Parameters
        ----------
        G : networkx graph
            Graph to simulate epidemic.
        infected_nodes : list
            List of infected nodes.
        beta : float
            Likelihood that the susceptible will be infected.
        mu : float
            Likelihood that the an individual will recover from infection.
        social_distancing : float
            Percentage of the population who practices social distancing.
        vaccinated_nodes : list, optional
            List of vaccinated nodes.
        vaccine_efficacy : float, optional
            Efficacy of the vaccine.
        seed : int, optional
            random seed

        Returns
        -------
        None.

        """
        SpreadingPhenomena.__init__(self, G, seed)
        self.__infected_nodes = infected_nodes
        self.__beta = beta
        self.__mu = mu
        self.__social_distancing = social_distancing
        self.__vaccinated_nodes = vaccinated_nodes
        self.__vaccine_efficacy = vaccine_efficacy

        node_status = {}
        for node in G.nodes():
            if node in self.__infected_nodes:
                node_status[node] = 1
            elif node in self.__vaccinated_nodes:
                node_status[node] = 3
            else:
                node_status[node] = 0
        nx.set_node_attributes(self._G, node_status, name='status')

    @property
    def infected_nodes(self):
        """Get list of infected nodes."""
        return list(filter(lambda x: x[1]['status'] == 1,
                           self._G.nodes(data=True)))

    def iterate(self, n):
        """
        Iterate n number of spreading phenomena using SIR model.

        Status = {0:'Susceptible', 1:'Infected', 2:'Recovered', 3:'Vaccinated'}

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
                if (((G.nodes[neighbor_nodes]['status'] == 0) or
                     (G.nodes[neighbor_nodes]['status'] == 3 and
                      np.random.random() > self.__vaccine_efficacy)) and
                        (np.random.random() >= self.__social_distancing)):
                    # If the neighbor is Susceptible or Vaccinated
                    # it has a beta probability
                    # and social_distancing probaility of getting infected
                    new_status = np.random.choice([0, 1], p=[1-self.__beta,
                                                             self.__beta])
                    G.nodes[neighbor_nodes]['status'] = new_status
            if G.nodes[node]['status'] == 1:
                # If infected, check the probability of recovery
                new_status = np.random.choice([1, 2], p=[1-self.__mu,
                                                         self.__mu])
                G.nodes[node]['status'] = new_status
        return G

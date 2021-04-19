# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:13:09 2021

@author: abelj
"""
import random
import networkx as nx
import numpy as np


class SpreadingPhenomena():
    """Class representation of Spreading Phenomena in the network."""

    def __init__(self, G, seed=None):
        """
        Initialize a new instance of the Spreading Phenomena.

        Parameters
        ----------
        G : networkx graph
            network to simulate spreading phenomena.
        seed : int, optional
            random seed

        Returns
        -------
        None.

        """
        self._G = G.copy()
        random.seed(seed)

    @property
    def G(self):
        """Return the networkx graph."""
        return self._G

    def iterate(self, n):
        """
        Iterate n number of spreading phenomena.

        Parameters
        ----------
        n : int
            number of iterations in the spreading phenomena.

        Returns
        -------
        graph : networkx graph
            yield graph of spreaded phenomena for each iteration

        """
        pass

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:41:07 2021.

@author: abelj, reference nxtools by Leodegario Lorenzo II
"""

import numpy as np


class NodeSizeScaler():
    """Node size scaler object."""

    def __init__(self, min_size=None, max_size=300):
        """
        Initiialize the node size scaler object.

        Parameters
        ----------
        min_size : float, default=None
            Minimum node size

        max_size : float, default=300
            Maximum node size
        """
        # Store minimum and maximum size
        self.min_size = min_size
        self.max_size = max_size

        # Initialize scale
        self.scale = None

    def fit(self, G):
        """
        Fit the scaler on a given networkx graph.

        Solves the required scale for the scaler.

        Parameters
        ----------
        G : networkx Graph
            Networkx graph to be used as reference for scaling
        """
        # Get the minimum and maximum degrees on the nodes of the graph
        degrees = G.degree()
        self.max_degree = max(dict(degrees).values())
        self.min_degree = min(dict(degrees).values())

        # Set min if none
        if self.min_size is None:
            self.min_size = self.max_size / self.max_degree * self.min_degree

        # Solve for the scale
        self.scale = ((self.max_size - self.min_size)
                      / (self.max_degree - self.min_degree))

    def transform(self, G) -> np.ndarray:
        """
        Return scaled node sizes for a given graph after fitting.

        Parameters
        ----------
        G : networkx Graph
            Networkx graph to be scaled

        Returns
        -------
        node_sizes : numpy array
            Node sizes array
        """
        # Scaler should have been fitted
        assert self.scale is not None, "Scaler should have been fitted."

        # Get node degrees as array
        degrees = np.array(list(dict(G.degree()).values()))

        # Compute for node sizes
        node_sizes = self.scale*(degrees - self.min_degree) + self.min_size

        return node_sizes

    def fit_transform(self, G) -> np.ndarray:
        """
        Return scaled node sizes for a given graph.

        Performs scale fitting then prediction.

        Parameters
        ----------
        G : networkx Graph
            Networkx graph to be scaled

        Returns
        -------
        node_sizes : numpy array
            Node sizes array
        """
        self.fit(G)
        node_sizes = self.transform(G)

        return node_sizes

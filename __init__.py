# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:49:16 2021

@author: abelj
"""
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from .NetworkProperties import NetworkProperties


def offset_pos(pos, dx, dy, nodes_to_offset=None):
    """
    Return offset node positions given dx and dy.

    Can be specified to offset specific nodes.

    Parameters
    ----------
    pos : dict
        Position dictionary with nodes as key, and position as value

    dx : float
        x-offset value

    dy : float
        y-offset value

    nodes_to_offset : list, default=None
        List of nodes to offset

    Returns
    -------
    new_pos : dict
        New node positions
    """
    # Initialize results container
    new_pos = {}

    # Iterate through all positions
    for node in pos.keys():
        # Offset only on the offset list if specified
        if (nodes_to_offset is not None) and (node not in nodes_to_offset):
            new_pos[node] = pos[node]
        else:
            new_pos[node] = pos[node] + np.array([dx, dy])

    return new_pos


def plot_degree_distribution(G, ax):
    """
    Plot degree distribution of networkx graph.

    Parameters
    ----------
    G : networkx graph
        Graph to plot degree distribution.
    ax : AxesSubplot:
        Place plot centrality on ax.

    Returns
    -------
    ax : AxesSubplot
        axes subplot of the histogram.
    """
    degrees = [k for node, k in nx.degree(G)]
    sns.histplot(degrees, ax=ax)
    return ax


def plot_centrality(G, centralities, ax, pos=None):
    """
    Plot centrality measure.

    Parameters
    ----------
    G : networkx graph
        Graph to plot centrality.
    centralities : array-like
        array of centralities, used for node color.
    ax : AxesSubplot:
        Place plot centrality on ax.
    pos : dict, optional
        key as node, value as xy coordinates.
        default fruchterman_reingold_layout.

    Returns
    -------
    ax, cb : AxesSubplot, ColorBar
        2-tuple of axes subplot of the plot and color bar.
    """
    if pos is None:
        pos = nx.fruchterman_reingold_layout(G, seed=0)
    cb = nx.draw_networkx_nodes(G, pos, node_size=300,
                                cmap=plt.cm.RdYlBu_r,
                                node_color=centralities,
                                ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', ax=ax)
    ax.set_axis_off()
    return ax, cb


def plot_all_centrality(G, pos=None):
    """Plot degree centrality."""
    fig = plt.figure(figsize=(20, 15))

    titles = ['Degree Centrality', 'Closeness Centrality',
              'Betweenness Centrality', 'Eigenvector Centrality']
    plots = [plot_degree_centrality, plot_closeness_centrality,
             plot_betweenness_centrality, plot_eigenvector_centrality]
    if pos is None:
        pos = nx.fruchterman_reingold_layout(G)

    for i, plot in enumerate(plots):
        ax = fig.add_subplot(2, 2, i+1)
        ax, cb = plot(G, ax, pos)
        fig.colorbar(cb, ax=ax)
        plt.title(titles[i])

    plt.show()


def plot_degree_centrality(G, ax, pos=None):
    """
    Plot degree centrality.

    Parameters
    ----------
    G : networkx graph
        Graph to plot centrality.
    ax : AxesSubplot:
        Place plot centrality on ax.
    pos : dict, optional
        key as node, value as xy coordinates.
        default fruchterman_reingold_layout.

    Returns
    -------
    ax, cb : AxesSubplot, ColorBar
        2-tuple of axes subplot of the plot and color bar.
    """
    centralities = list(nx.degree_centrality(G).values())
    return plot_centrality(G, centralities, ax, pos)


def plot_closeness_centrality(G, ax, pos=None):
    """
    Plot closeness centrality.

    Parameters
    ----------
    G : networkx graph
        Graph to plot centrality.
    ax : AxesSubplot:
        Place plot centrality on ax.
    pos : dict, optional
        key as node, value as xy coordinates.
        default fruchterman_reingold_layout.

    Returns
    -------
    ax, cb : AxesSubplot, ColorBar
        2-tuple of axes subplot of the plot and color bar.
    """
    centralities = list(nx.closeness_centrality(G).values())
    return plot_centrality(G, centralities, ax, pos)


def plot_betweenness_centrality(G, ax, pos=None):
    """
    Plot betweenness centrality.

    Parameters
    ----------
    G : networkx graph
        Graph to plot centrality.
    ax : AxesSubplot:
        Place plot centrality on ax.
    pos : dict, optional
        key as node, value as xy coordinates.
        default fruchterman_reingold_layout.

    Returns
    -------
    ax, cb : AxesSubplot, ColorBar
        2-tuple of axes subplot of the plot and color bar.
    """
    centralities = list(nx.betweenness_centrality(G).values())
    return plot_centrality(G, centralities, ax, pos)


def plot_eigenvector_centrality(G, ax, pos=None):
    """
    Plot eigenvector centrality.

    Parameters
    ----------
    G : networkx graph
        Graph to plot centrality.
    ax : AxesSubplot:
        Place plot centrality on ax.
    pos : dict, optional
        key as node, value as xy coordinates.
        default fruchterman_reingold_layout.

    Returns
    -------
    ax, cb : AxesSubplot, ColorBar
        2-tuple of axes subplot of the plot and color bar.
    """
    centralities = list(nx.eigenvector_centrality(G).values())
    return plot_centrality(G, centralities, ax, pos)


def connected_component_subgraphs(G):
    """
    Get connected component subgraphs of networkx G.

    Parameters
    ----------
    G : networkx graph
        Graph to get connected components.

    Returns
    -------
    connected_component_subgraphs : array like
        List of networkx graph of each component in G.

    """
    return [G.subgraph(c).copy() for c in nx.connected_components(G)]


def compare_graphs(graphs):
    """
    Compare avg. clustering coefficient and avg. shortest path length
    of the graphs.


    Parameters
    ----------
    graphs : list of 2 tuples of name of the graph and networkx graphs
        List of networkx graphs to compare.

    Returns
    -------
    comparison : pandas DataFrame
        Pandas DataFrame where index is the name of the model and columns are
        C = average clustering coefficient.
        l = average shortest path length.

    """
    df_result = pd.DataFrame(columns=['Name', '$C$', '$l$'])

    for name, graph in graphs:
        net_prop = NetworkProperties(graph)
        avg_coeff = net_prop.avg_clustering_coeff
        avg_shortest_path = net_prop.avg_shortest_path
        df_result = df_result.append({'Name': name, '$C$': avg_coeff,
                                      '$l$': avg_shortest_path},
                                     ignore_index=True)

    return df_result.set_index('Name')

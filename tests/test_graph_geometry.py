import numpy as np
import pandas as pd

from deepform.data.graph_geometry import document_edges

# ASCII Art of Test Example
#
# A --- B --- C
# |  W--|--X  |
# D -|- E -|- F
# |  Y -|- Z  |
# G --- H --- I

A = {"token": "A", "x0": 1, "y1": 1}
B = {"token": "B", "x0": 3, "y1": 1}
C = {"token": "C", "x0": 5, "y1": 1}
D = {"token": "D", "x0": 1, "y1": 3}
E = {"token": "E", "x0": 3, "y1": 3}
F = {"token": "F", "x0": 5, "y1": 3}
G = {"token": "G", "x0": 1, "y1": 5}
H = {"token": "H", "x0": 3, "y1": 5}
I = {"token": "I", "x0": 5, "y1": 5}  # noqa: E741
W = {"token": "W", "x0": 2, "y1": 2}
X = {"token": "X", "x0": 4, "y1": 2}
Y = {"token": "Y", "x0": 2, "y1": 4}
Z = {"token": "Z", "x0": 4, "y1": 4}

tokens = pd.DataFrame.from_records([A, B, C, D, E, F, G, H, I, W, X, Y, Z])

# Manually construct the sparse matrix of edges for the above example.
edges = np.zeros((13, 13))
edges[0, 1] = 1  # A B
edges[1, 2] = 1  # B C
edges[3, 4] = 1  # D E
edges[4, 5] = 1  # E F
edges[6, 7] = 1  # G H
edges[7, 8] = 1  # H I
edges[0, 3] = 1  # A D
edges[3, 6] = 1  # D G
edges[1, 4] = 1  # B E
edges[4, 7] = 1  # E H
edges[2, 5] = 1  # C F
edges[5, 8] = 1  # F I
edges[9, 10] = 1  # W X
edges[11, 12] = 1  # Y Z
edges[9, 11] = 1  # W Y
edges[10, 12] = 1  # X Z

# Add in the symmetric relationships
edges = edges + edges.T

adjacency = document_edges(tokens)
expected = edges + np.eye(13)


def test_9x9_adjacency():
    adjacency9x9 = adjacency[0:9, 0:9]
    expected9x9 = expected[0:9, 0:9]
    assert (adjacency9x9 == expected9x9).all()


def test_4x4_adjacency():
    adjacency4x4 = adjacency[9:, 9:]
    expected4x4 = expected[9:, 9:]
    assert (adjacency4x4 == expected4x4).all()


def test_disconnected():
    disconnectedRight = adjacency[9:, 0:9]
    disconnectedBottom = adjacency[0:9, 9:]
    assert (disconnectedRight == 0).all()
    assert (disconnectedBottom == 0).all()

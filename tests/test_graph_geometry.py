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

A = {"token": "A", "x0": 1, "y1": 1, "page": 0}
B = {"token": "B", "x0": 3, "y1": 1, "page": 0}
C = {"token": "C", "x0": 5, "y1": 1, "page": 0}
D = {"token": "D", "x0": 1, "y1": 3, "page": 0}
E = {"token": "E", "x0": 3, "y1": 3, "page": 0}
F = {"token": "F", "x0": 5, "y1": 3, "page": 0}
G = {"token": "G", "x0": 1, "y1": 5, "page": 0}
H = {"token": "H", "x0": 3, "y1": 5, "page": 0}
I = {"token": "I", "x0": 5, "y1": 5, "page": 0}  # noqa: E741
W = {"token": "W", "x0": 2, "y1": 2, "page": 0}
X = {"token": "X", "x0": 4, "y1": 2, "page": 0}
Y = {"token": "Y", "x0": 2, "y1": 4, "page": 0}
Z = {"token": "Z", "x0": 4, "y1": 4, "page": 0}

tokens = pd.DataFrame.from_records([A, B, C, D, E, F, G, H, I, W, X, Y, Z])

# Manually construct the sparse matrix of edges for the above example.
edges = np.zeros((13, 13))
edges[0, 1] = True  # A B
edges[1, 2] = True  # B C
edges[3, 4] = True  # D E
edges[4, 5] = True  # E F
edges[6, 7] = True  # G H
edges[7, 8] = True  # H I
edges[0, 3] = True  # A D
edges[3, 6] = True  # D G
edges[1, 4] = True  # B E
edges[4, 7] = True  # E H
edges[2, 5] = True  # C F
edges[5, 8] = True  # F I
edges[9, 10] = True  # W X
edges[11, 12] = True  # Y Z
edges[9, 11] = True  # W Y
edges[10, 12] = True  # X Z

# Add in the symmetric relationships
edges = edges + edges.T

adjacency = document_edges(tokens).todense()
expected = edges


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


def test_different_pages():
    B_pg_2 = B.copy()
    B_pg_2["page"] = 1
    tokens_pages = pd.DataFrame.from_records([A, B_pg_2, C])

    adjacency = document_edges(tokens_pages).todense()
    assert not adjacency[0, 1]
    assert adjacency[0, 2]

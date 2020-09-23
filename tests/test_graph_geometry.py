import numpy as np
import pandas as pd

from deepform.data.graph_geometry import document_edges

# ASCII Art of Test Example
#
# A ---------- B
# |            |
# |     E      |
# |            |
# C ---------- D

pt_A = {"token": "A", "x0": 10, "y1": 10}
pt_B = {"token": "B", "x0": 20, "y1": 10}
pt_C = {"token": "C", "x0": 10, "y1": 20}
pt_D = {"token": "D", "x0": 20, "y1": 20}
pt_E = {"token": "E", "x0": 15, "y1": 15}
tokens = pd.DataFrame.from_records([pt_A, pt_B, pt_C, pt_D, pt_E])


def test_adjacency_matrix():
    adjacency = document_edges(tokens)
    edges = np.matrix(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = edges + np.eye(5)
    assert (adjacency == expected).all()

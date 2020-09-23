import numpy as np
import numpy.ma as ma


def document_edges(tokens, relative_tolerance=0.01):
    """"""

    # For now we compute alignment of text tokens based on their lower left corner.
    dX = np.subtract.outer(tokens["x0"].to_numpy(), tokens["x0"].to_numpy())

    dY = np.subtract.outer(tokens["y1"].to_numpy(), tokens["y1"].to_numpy())

    D = np.abs(dX) + np.abs(dY)
    V_sim = dY / D
    H_sim = dX / D

    dX_h_aligned = ma.masked_where(
        np.logical_not(np.isclose(np.abs(H_sim), 1, rtol=relative_tolerance)), dX
    )
    dY_v_aligned = ma.masked_where(
        np.logical_not(np.isclose(np.abs(V_sim), 1, rtol=relative_tolerance)), dY
    )

    # TODO: Integrate filtering for direct neighbors

    # right_masked = ma.masked_where(np.less(dX_h_aligned, 0), dX_h_aligned)
    # test_right = np.argmin(right_masked, axis=0)
    #
    # print(right_masked)
    # print(test_right)
    # print(right_masked.shape)
    #
    # print(right_masked)
    #
    # test_left = np.argmax(
    #     ma.masked_where(np.greater(dX_h_aligned, 0), dX_h_aligned), axis=0
    # )
    # test_bottom = np.argmin(
    #     ma.masked_where(np.less(dY_v_aligned, 0), dY_v_aligned), axis=0
    # )
    # test_top = np.argmax(
    #     ma.masked_where(np.greater(dY_v_aligned, 0), dY_v_aligned), axis=0
    # )
    #
    # print(test_left)
    # print(test_top)
    # print(test_bottom)

    aligned = np.logical_xor(dX_h_aligned.mask, dY_v_aligned.mask)
    adjacency = np.zeros(D.shape)
    adjacency[aligned] = 1
    adjacency = np.eye(D.shape[0]) + adjacency

    return adjacency

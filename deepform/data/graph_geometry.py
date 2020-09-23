import numpy as np
import numpy.ma as ma
<<<<<<< HEAD
import scipy.sparse as sparse
=======
>>>>>>> 45dcc52... Add basic adjacench matrix function for tokens with test


def document_edges(tokens, relative_tolerance=0.01):
    """"""
<<<<<<< HEAD
    N = len(tokens)

    # For now we compute alignment of text tokens based on their lower left corner.
    dX = np.subtract.outer(tokens["x0"].to_numpy(), tokens["x0"].to_numpy())
    dY = np.subtract.outer(tokens["y1"].to_numpy(), tokens["y1"].to_numpy())
    page_mask = np.not_equal.outer(tokens["page"].to_numpy(), tokens["page"].to_numpy())
=======

    # For now we compute alignment of text tokens based on their lower left corner.
    dX = np.subtract.outer(tokens["x0"].to_numpy(), tokens["x0"].to_numpy())

    dY = np.subtract.outer(tokens["y1"].to_numpy(), tokens["y1"].to_numpy())
>>>>>>> 45dcc52... Add basic adjacench matrix function for tokens with test

    D = np.abs(dX) + np.abs(dY)
    V_sim = dY / D
    H_sim = dX / D

    dX_h_aligned = ma.masked_where(
<<<<<<< HEAD
        np.logical_or(
            page_mask,
            np.logical_not(np.isclose(np.abs(H_sim), 1, rtol=relative_tolerance)),
        ),
        dX,
    )
    dY_v_aligned = ma.masked_where(
        np.logical_or(
            page_mask,
            np.logical_not(np.isclose(np.abs(V_sim), 1, rtol=relative_tolerance)),
        ),
        dY,
    )

    test_right = ma.masked_where(np.greater(dX_h_aligned, 0), dX_h_aligned)
    test_bottom = ma.masked_where(np.greater(dY_v_aligned, 0), dY_v_aligned)

    right_max = np.argmax(test_right, axis=0)
    bottom_max = np.argmax(test_bottom, axis=0)

    adjacency = sparse.lil_matrix((N, N), dtype=np.bool_)

    for i in range(len(tokens)):
        if dX_h_aligned[i, right_max[i]]:
            adjacency[i, right_max[i]] = True
            adjacency[right_max[i], i] = True
        if dY_v_aligned[i, bottom_max[i]]:
            adjacency[i, bottom_max[i]] = True
            adjacency[bottom_max[i], i] = True

    return adjacency.tocoo()
=======
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
>>>>>>> 45dcc52... Add basic adjacench matrix function for tokens with test

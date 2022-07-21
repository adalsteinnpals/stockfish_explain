from utils import get_saliency_mat
import chess


non_king_pieces = ["p", "b", "n", "r", "q"]


def test_saliency():
    board = chess.Board("3q2k1/5b1p/1pr2pp1/p2p4/3P3P/1P1NQP2/P2R2K1/8 w - - 0 1")
    method = "shapley"
    perturb_pieces = non_king_pieces

    saliency = get_saliency_mat(board, perturb_pieces, method, n_samples=5, color=None)

    assert len(saliency["mat"][0]) == 17

    saliency = get_saliency_mat(
        board, perturb_pieces, method, n_samples=5, color="white"
    )

    assert len(saliency["mat"][0]) == 8

    saliency = get_saliency_mat(
        board, perturb_pieces, method, n_samples=5, color="black"
    )

    assert len(saliency["mat"][0]) == 9


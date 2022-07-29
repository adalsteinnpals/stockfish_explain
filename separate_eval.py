from io import BytesIO

import cairosvg
import chess
import chess.svg
import click
import matplotlib.pyplot as plt
import numpy as np
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import Lime, ShapleyValueSampling
from captum.attr._core.lime import get_exp_kernel_similarity_function
from PIL import Image

from utils import (
    available_methods,
    eval_class,
    get_nnue_eval_from_fen,
    get_saliency_mat,
    non_king_pieces,
    remove_kings_from_piece_map,
)


@click.command()
@click.option(
    "--fen_string",
    prompt="Fen string",
    help="The fen string to use.",
    default="8/1N4k1/7p/4ppbP/4P3/1PP5/1K6/8",
)
@click.option(
    "--include_pieces",
    prompt="Perturb pieces (e.g. A for all pieces, NB for knight and bishop, etc.)",
    help="The pieces to use in perturbation.",
    default="A",
)
@click.option(
    "--method",
    prompt="Perturbation method:",
    default="shapley",
    help="The pieces to use in perturbation.",
)
@click.option(
    "--separate",
    prompt="Evaluate values for each player separately",
    default="Y",
    help="The pieces to use in perturbation.",
)
@click.option(
    "--ending",
    prompt="Add string to file name: ",
    default="",
    help="The pieces to use in perturbation.",
)
def main(fen_string, include_pieces, method, separate, ending):

    fig = main_separate(fen_string, include_pieces, method, separate)
    
    plt.savefig(f"figures/heatmap_{ending}.pdf")


def main_separate(fen_string, include_pieces, method, separate):

    if method not in available_methods:
        raise ValueError(f"Method value [{method}] not in {available_methods}")

    board = chess.Board(fen_string)

    if include_pieces.lower() == "a":
        perturb_pieces = non_king_pieces
    else:
        perturb_pieces = list(include_pieces.lower())
        assert all([p in non_king_pieces for p in perturb_pieces])

    if separate == "Y":
        saliency = get_saliency_mat(board, perturb_pieces, method, color="white")
        white_mat = saliency["board_mat"]
        saliency = get_saliency_mat(board, perturb_pieces, method, color="black")
        black_mat = saliency["board_mat"]
        board_mat = white_mat + black_mat
    elif separate == "N":
        saliency = get_saliency_mat(board, perturb_pieces, method, color=None)
        board_mat = saliency["board_mat"]
    else:
        raise ValueError(f"Expected separate to be Y or N, got {separate}")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))

    max_abs_val = np.abs(board_mat).max()
    ax2_plot = ax2.imshow(board_mat, vmin=-max_abs_val, vmax=max_abs_val, cmap="bwr")

    ax2.hlines(
        y=np.arange(0, 8) + 0.5,
        xmin=np.full(8, 0) - 0.5,
        xmax=np.full(8, 8) - 0.5,
        color="black",
    )
    ax2.vlines(
        x=np.arange(0, 8) + 0.5,
        ymin=np.full(8, 0) - 0.5,
        ymax=np.full(8, 8) - 0.5,
        color="black",
    )
    ax2.set_xticks(list(range(8)))
    ax2.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax2.set_yticks(list(range(8)))
    ax2.set_yticklabels(["8", "7", "6", "5", "4", "3", "2", "1"])

    ax2.set_title(f"Method: {method}, Pieces: {include_pieces}, Value: {saliency['value'].item()}")

    plt.colorbar(ax2_plot, ax=ax2)

    for i in range(board_mat.shape[0]):
        for j in range(board_mat.shape[1]):
            c = board_mat[j, i]
            if c != 0:
                ax2.text(
                    i, j, f"{c:.2f}", va="center", ha="center", c="black", fontsize=20
                )

    svg = chess.svg.board(board)
    img = cairosvg.svg2png(svg)
    img = Image.open(BytesIO(img))
    ax1_plot = ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()

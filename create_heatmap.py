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
    eval_class,
    get_nnue_eval_from_fen,
    get_saliency_mat,
    remove_kings_from_piece_map,
    available_methods,
    non_king_pieces,
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
def main(fen_string, include_pieces, method):

    if method not in available_methods:
        raise ValueError(f"Method value [{method}] not in {available_methods}")

    board = chess.Board(fen_string)

    if include_pieces.lower() == "a":
        perturb_pieces = non_king_pieces
    else:
        perturb_pieces = list(include_pieces.lower())
        assert all([p in non_king_pieces for p in perturb_pieces])

    mat, chosen_map_keys = get_saliency_mat(board, perturb_pieces, method)

    zeros_mat = np.zeros((8, 8))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))

    for i in range(mat.shape[1]):
        zeros_mat.ravel()[63 - chosen_map_keys[i]] = mat[0, i]
    zeros_mat = np.fliplr(zeros_mat)
    ax2_plot = ax2.imshow(zeros_mat)
    plt.colorbar(ax2_plot, ax=ax2)

    for i in range(zeros_mat.shape[0]):
        for j in range(zeros_mat.shape[1]):
            c = zeros_mat[j, i]
            if c != 0:
                ax2.text(i, j, f"{c:.2f}", va="center", ha="center", c="black")

    svg = chess.svg.board(board)
    img = cairosvg.svg2png(svg)
    img = Image.open(BytesIO(img))
    ax1_plot = ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    plt.savefig("heatmap.pdf")


if __name__ == "__main__":
    main()

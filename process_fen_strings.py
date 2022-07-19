from io import BytesIO

import cairosvg
import chess
import chess.svg
import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import (
    available_methods,
    get_saliency_mat,
    non_king_pieces,
)


@click.command()
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
def main(include_pieces, method):

    with open("fen_strings.txt") as f:
        fen_strings = f.readlines()

    num_fen_strings = len(fen_strings)

    fen_string = fen_strings[0]

    if method not in available_methods:
        raise ValueError(f"Method value [{method}] not in {available_methods}")

    if include_pieces.lower() == "a":
        perturb_pieces = non_king_pieces
    else:
        perturb_pieces = list(include_pieces.lower())
        assert all([p in non_king_pieces for p in perturb_pieces])

    mats = []
    plt.figure(figsize=(20, 10 * num_fen_strings))

    for idx, fen_string in enumerate(fen_strings):
        board = chess.Board(fen_string)

        mat, chosen_map_keys = get_saliency_mat(board, perturb_pieces, method)

        board_mat = np.zeros((8, 8))
        for i in range(mat.shape[1]):
            board_mat.ravel()[63 - chosen_map_keys[i]] = mat[0, i]
        board_mat = np.fliplr(board_mat)

        ax1 = plt.subplot(num_fen_strings, 2, idx * 2 + 1)
        ax2 = plt.subplot(num_fen_strings, 2, idx * 2 + 2)

        ax2_plot = ax2.imshow(board_mat)
        plt.colorbar(ax2_plot, ax=ax2)

        for i in range(board_mat.shape[0]):
            for j in range(board_mat.shape[1]):
                c = board_mat[j, i]
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

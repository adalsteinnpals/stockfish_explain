from io import BytesIO

import cairosvg
import chess
import chess.svg
import click
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import ShapleyValueSampling
from PIL import Image

from utils import (eval_class, get_nnue_eval_from_fen,
                   remove_kings_from_piece_map)

minor_pieces = ['p','b','n','r', 'q']


@click.command()
@click.option('--fen_string', prompt='Fen string',
              help='The fen string to use.')
@click.option('--include_pieces', prompt='Perturb pieces (e.g. A for all pieces, NB for knight and bishop, etc.)',
              help='The pieces to use in perturbation.')

def main(fen_string, include_pieces):

    board = chess.Board(fen_string)

    if include_pieces.lower() == 'a':
        perturb_pieces = minor_pieces
    else:
        perturb_pieces = list(include_pieces.lower())
        assert all([p in minor_pieces for p in perturb_pieces])

        

    eval = eval_class(board, pertub_pieces = perturb_pieces)
    alg_svs = ShapleyValueSampling(eval)

    mat =  alg_svs.attribute(eval.input,
                            baselines = 0,
                            target=0,
                            perturbations_per_eval = 1,
                            n_samples=50,
                            show_progress=True).detach().cpu().numpy()




    zeros_mat = np.zeros((8,8))
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20,10))

    for i in range(mat.shape[1]):
        zeros_mat.ravel()[63-eval.chosen_map_keys[i]] = mat[0,i]
    zeros_mat = np.fliplr(zeros_mat)
    ax2_plot = ax2.imshow(zeros_mat)
    plt.colorbar(ax2_plot, ax=ax2)


    for i in range(zeros_mat.shape[0]):
        for j in range(zeros_mat.shape[1]):
            c = zeros_mat[j, i]
            if c != 0:
                ax2.text(i, j, f'{c:.2f}', va='center', ha='center', c ='black')


    svg = chess.svg.board(board)
    img = cairosvg.svg2png(svg)
    img = Image.open(BytesIO(img))
    ax1_plot = ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.tight_layout()
    plt.savefig('heatmap.pdf')


if __name__ == '__main__':
    main()

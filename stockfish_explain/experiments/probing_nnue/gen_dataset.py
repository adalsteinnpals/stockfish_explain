
import torch
import pandas as pd
import chess
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from stockfish_explain.utils.general import   get_FenBatchProvider, transform
from stockfish_explain.gen_concepts import create_custom_concepts
import logging

# set default plot size as large
plt.rcParams['figure.figsize'] = [20, 10]

# initialize logger
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

def create_concepts(dataset_size = 200000):

    batch_size = 50
    num_batch = dataset_size // batch_size

    logging.warning("Creating concepts for dataset of size {}".format(dataset_size))

    FBP = get_FenBatchProvider(batch_size=batch_size)

    def fetch_batch():
        fens = next(FBP)
        results = []
        for i, fen in enumerate(fens):
            d = {}
            d['fen'] = fen
            board = chess.Board(fen)
            pieces = dict(Counter([str(v) for v in board.piece_map().values()]))
            d = {**d, **pieces}

            results.append(d)
        return results


    data = []
    for i in tqdm(range(num_batch)):
        data = data  + fetch_batch()

    df = pd.DataFrame(data).fillna(0)
    del data
    logging.info("dataframe size: {}".format(df.shape))



    # add concept_dict to df
    for idx, row in tqdm(df.iterrows()):
        board = chess.Board(row['fen'])
        concept_dict = create_custom_concepts(board)
        for k, v in concept_dict.items():
            df.at[idx, k] = v   

    logging.info("dataframe size after concepts: {}".format(df.shape))

    columns_names = {
        'K': 'white_king',
        'k': 'black_king',
        'Q': 'white_queen',
        'q': 'black_queen',
        'R': 'white_rook',
        'r': 'black_rook',
        'B': 'white_bishop',
        'b': 'black_bishop',
        'N': 'white_knight',
        'n': 'black_knight',
        'P': 'white_pawn',
        'p': 'black_pawn',
    }


    df = df.rename(columns=columns_names)

    return df


if __name__ == '__main__':
    create_concepts(dataset_size = 2000)

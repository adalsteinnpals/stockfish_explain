
import torch
import pandas as pd
import chess
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from utils import   get_FenBatchProvider, transform
from stockfish_explain.gen_concepts import create_custom_concepts
import sqlite3

def main(num_samples = 150000):
    batch_size = 50
    num_batch = num_samples // batch_size

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
    print(f'Len data: {len(data)}')

    df = pd.DataFrame(data).fillna(0)
    del data
    print(f'df shape: {df.shape}')

    print(df.shape)
    # add concept_dict to df
    for idx, row in tqdm(df.iterrows()):
        board = chess.Board(row['fen'])
        concept_dict = create_custom_concepts(board)
        for k, v in concept_dict.items():
            df.at[idx, k] = v   
    print(df.shape)

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

    conn = sqlite3.connect('chess_auto_encoder.db')
    df.to_sql('fen_concept_df', conn, if_exists='replace', index=True)
    conn.close()

if __name__ == '__main__':
    num_samples = 200000
    main(num_samples=num_samples)
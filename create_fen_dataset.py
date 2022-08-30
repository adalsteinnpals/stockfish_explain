from stockfish_project.nnue_pytorch.nnue_dataset import FenBatchProvider
from stockfish_project.nnue_pytorch.gen_dataset import get_nnue_eval_from_fen
from tqdm import tqdm 
import chess
import pandas as pd
import copy
from collections import Counter



def remove_pieces_from_piece_map(piece_map, pieces):
    _piece_map = copy.copy(piece_map)
    for key, value in piece_map.items():
        if str(value) in pieces:
            del _piece_map[key]

    return _piece_map




def gen_fen_dataset():
    batch_size = 100
    num_batch = 10
    FBP = FenBatchProvider('../data/training_data.binpack', True, 1, batch_size=batch_size)


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
    

    df_filtered = (
        df
        .query('(b == 1 & n == 0 & N == 1 & B == 0) | (b == 0 & n == 1 & N == 0 & B == 1)')
    )

    results = []
    for idx, row in tqdm(df_filtered.iterrows()):
        _, eval_dict = get_nnue_eval_from_fen(row['fen'])
        if eval_dict is not None:
            results.append({**row, **eval_dict})
    df_filtered = pd.DataFrame(results)

    print(df_filtered.shape)
    print(df_filtered.head(10))


    results = []
    for idx, row in tqdm(df_filtered.iterrows()):
        res_ = {}
        if (row['n'] == 1) & (row['B'] == 1):
            board = chess.Board(row['fen'])
            org_piece_map = board.piece_map()

            # Remove both
            piece_map = remove_pieces_from_piece_map(org_piece_map, ['n' , 'B'])
            board.set_piece_map(piece_map)
            _, eval_dict = get_nnue_eval_from_fen(board.fen())
            if eval_dict is not None:
                res_['remove_both'] = eval_dict['NNUE']

                piece_map = remove_pieces_from_piece_map(org_piece_map, ['B'])
                board.set_piece_map(piece_map)
                _, eval_dict = get_nnue_eval_from_fen(board.fen())
                if eval_dict is not None:
                    res_['remove_B'] = eval_dict['NNUE']

                    piece_map = remove_pieces_from_piece_map(org_piece_map, ['n'])
                    board.set_piece_map(piece_map)
                    _, eval_dict = get_nnue_eval_from_fen(board.fen())
                    if eval_dict is not None:
                        res_['remove_n'] = eval_dict['NNUE']

                        results.append({**row, **res_})

    df_filtered = pd.DataFrame(results)
    print(df_filtered.shape)
    print(df_filtered.head(10))

if __name__ == '__main__':
    gen_fen_dataset()


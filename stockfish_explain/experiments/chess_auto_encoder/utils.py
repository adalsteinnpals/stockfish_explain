from stockfish_project.nnue_pytorch.halfkp import Features
from stockfish_project.nnue_pytorch.nnue_dataset import FenBatchProvider
import chess
import torch


def get_FenBatchProvider(batch_size=100):
    return FenBatchProvider('../../../../data/training_data.binpack', True, 1, batch_size=batch_size)


halfkp_features = Features()

def transform(batch):
    halfkp_list = []
    for fen in batch:
        board = chess.Board(fen)
        features = halfkp_features.get_active_features(board)
        halfkp_list.append(torch.cat(features))

    return torch.stack(halfkp_list)

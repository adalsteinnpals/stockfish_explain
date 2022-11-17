#from stockfish_project.nnue_pytorch.halfkp import Features
from stockfish_explain.external_packages.nnue_pytorch.halfkp import Features as halfkp_features
from stockfish_explain.external_packages.nnue_pytorch.custom_features import Features as custom_features
from stockfish_explain.external_packages.nnue_pytorch.nnue_dataset import FenBatchProvider
import chess
import torch


def get_FenBatchProvider(batch_size=100):
    return FenBatchProvider('../../../../data/training_data.binpack', True, 1, batch_size=batch_size)


feature_method = custom_features()

def transform(batch):
    feature_list = []
    for fen in batch:
        board = chess.Board(fen)
        features = feature_method.get_active_features(board)
        feature_list.append(features[0])
        #halfkp_list.append(torch.cat(features))

    return torch.stack(feature_list)

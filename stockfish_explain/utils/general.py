
from stockfish_explain.external_packages.nnue_pytorch.halfkp import Features as halfkp_features
from stockfish_explain.external_packages.nnue_pytorch.custom_features import Features as custom_features
from stockfish_explain.external_packages.nnue_pytorch.nnue_dataset import FenBatchProvider
import chess
import torch
import matplotlib.pyplot as plt
import os

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_FenBatchProvider(batch_size=100):
    return FenBatchProvider(get_root_dir() + '/data/training_data.binpack', True, 1, batch_size=batch_size)


feature_method = custom_features()

def transform(batch):
    feature_list = []
    for fen in batch:
        board = chess.Board(fen)
        features = feature_method.get_active_features(board)
        feature_list.append(features[0])
        #halfkp_list.append(torch.cat(features))

    return torch.stack(feature_list)

def plot_results(df_results):
    for target_name in df_results.target_name.unique():
        df_results_ = df_results[df_results.target_name == target_name]
        for model_name  in df_results_.model_name.unique():
            # plot scores
            plt.plot(range(len(df_results_[df_results_.model_name == model_name])), df_results_[df_results_.model_name == model_name].score, label=model_name)

        # set x ticks as size   
        plt.xticks(range(len(df_results_[df_results_.model_name == model_name])), df_results_[df_results_.model_name == model_name]['size'].astype(str))
        plt.title(target_name)
        plt.grid()
        plt.ylabel('score')
        plt.xlabel('Encoder-Decoder compression size')
        plt.ylim(0,1.1)
        plt.legend()
        plt.show()





if __name__ == '__main__':
    print(get_root_dir())

    print(next(get_FenBatchProvider()))
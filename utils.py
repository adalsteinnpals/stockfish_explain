import copy
import re

import chess
import numpy as np
import pandas as pd
import torch
import yaml
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import Lime, ShapleyValueSampling
from captum.attr._core.lime import get_exp_kernel_similarity_function
from stockfish import Stockfish
from torch import nn

non_king_pieces = ["p", "b", "n", "r", "q"]
available_methods = ["shapley", "lime"]


def get_config():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = get_config()


class stockfish_eval(Stockfish):
    def get_evaluation_table(self, verbose=False):
        def get_evaluation_dict(text_list, version=14):
            eval_dict = {}
            for _text in text_list:
                _text = re.sub(" +", " ", _text)
                _text = re.sub(" side", "-side", _text)
                if len(_text) > 10:
                    _text_split = _text.split(" ")
                    if len(_text_split) > 1:
                        if version == 14:
                            if _text_split[1] == "evaluation":
                                eval_dict[_text_split[0]] = _text_split[2]
                                eval_dict["side"] = _text_split[3]
                        elif version == 8:
                            if _text_split[1] == "Evaluation:":
                                eval_dict["Classical"] = _text_split[2]
                                eval_dict["side"] = _text_split[3]
            return eval_dict

        fen_position = self.get_fen_position()

        self._put(f"{'eval'}")
        text = []
        while True:
            text_ = self._read_line()
            if verbose:
                print(text_)
            text.append(text_)
            if "Final evaluation" in text_:
                break
            if "Total Evaluation" in text_:
                break

        # if verbose: [print(text_) for text_ in text]
        # if verbose: print('len text: ', len(text))

        if len(text) == 21:
            start_idx = 5
            offset = 0
            eval_dict = get_evaluation_dict(text, version=8)
        elif len(text) == 72:
            start_idx = 9
            offset = 1
            eval_dict = get_evaluation_dict(text, version=14)
        else:
            return None, None

        t = []
        for i in range(start_idx, start_idx + offset + 14):
            if i == start_idx + offset + 12:
                continue
            text_split = text[i].split("|")
            new_text = []
            for text_ in text_split:
                new_text += (
                    text_.replace("King safety", "King_safety")
                    .replace("Passed pawns", "Passed_pawns")
                    .strip()
                    .split(" ")
                )
            new_text = [x for x in new_text if x != ""]
            t.append(new_text)

        df = pd.DataFrame(
            t,
            columns=[
                "Eval term",
                "MG_white",
                "EG_white",
                "MG_black",
                "EG_black",
                "MG_total",
                "EG_total",
            ],
        )
        df = df.set_index("Eval term")
        return df, eval_dict


def get_nnue_eval_from_fen(fen, parameters=None):
    stockfish = stockfish_eval(path=config["STOCKFISH_PATH"], parameters=parameters)
    stockfish.set_fen_position(fen)
    df, eval_dict = stockfish.get_evaluation_table(verbose=False)
    return {"table": df, "eval": eval_dict}


def remove_kings_from_piece_map(piece_map):
    _piece_map = copy.copy(piece_map)
    _kings_map = {}
    for key, value in piece_map.items():
        if str(value) in ["k", "K"]:
            del _piece_map[key]
            _kings_map[key] = value

    return _piece_map, _kings_map


def reduce_piece_map(map, list_pieces=["p", "b", "n", "r", "q"], color=None):
    _map = copy.copy(map)
    _chosen_map = {}
    for key, value in map.items():
        if str(value).lower() in list_pieces:
            if color is None:
                del _map[key]
                _chosen_map[key] = value
            elif (color == "white") and (str(value).isupper()):
                del _map[key]
                _chosen_map[key] = value
            elif (color == "black") and (str(value).islower()):
                del _map[key]
                _chosen_map[key] = value
    return _chosen_map, _map


class eval_class(nn.Module):
    def __init__(self, board, pertub_pieces=["p", "b", "n", "r", "q"], color=None):
        self.board = board
        self.pertub_pieces = pertub_pieces
        self.color = color
        super(eval_class, self).__init__()
        self.org_piece_map = board.piece_map()
        self.map, self.king_map = remove_kings_from_piece_map(self.org_piece_map)
        self.chosen_map, self.static_map = reduce_piece_map(
            self.map, list_pieces=self.pertub_pieces, color=self.color
        )

        self.chosen_map_keys = list(self.chosen_map.keys())
        self.input = torch.from_numpy(np.ones(len(self.chosen_map))).unsqueeze(0)

    def forward(self, x):
        """
        x is a binary input of all non-king pieces
        if x is all ones, position is full
        if x is all zeros, position is kings only


        """

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        _map = {}
        for idx in range(x.shape[1]):
            if x[0, idx] == 1:
                square = self.chosen_map_keys[idx]
                _map[square] = self.map[square]

        piece_map = {**_map, **self.static_map, **self.king_map}

        board = chess.Board()
        board.set_piece_map(piece_map)

        results = get_nnue_eval_from_fen(board.fen())

        return_value = (
            torch.from_numpy(np.array(float(results["eval"]["NNUE"])))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return return_value


def get_saliency_mat(board, perturb_pieces, method, n_samples=50, color=None):
    eval = eval_class(board, pertub_pieces=perturb_pieces, color=color)

    if method == "shapley":
        alg_svs = ShapleyValueSampling(eval)

        mat = (
            alg_svs.attribute(
                eval.input,
                baselines=0,
                target=0,
                perturbations_per_eval=1,
                n_samples=n_samples,
                show_progress=True,
            )
            .detach()
            .cpu()
            .numpy()
        )
    elif method == "lime":
        exp_eucl_distance = get_exp_kernel_similarity_function(
            "euclidean", kernel_width=1000
        )
        alg_lime = Lime(
            eval,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance,
        )

        mat = (
            alg_lime.attribute(
                eval.input,
                target=0,
                n_samples=n_samples,
                perturbations_per_eval=1,
                show_progress=True,
            )
            .detach()
            .cpu()
            .numpy()
        )

    board_mat = np.zeros((8, 8))
    for i in range(mat.shape[1]):
        board_mat.ravel()[63 - eval.chosen_map_keys[i]] = mat[0, i]
    board_mat = np.fliplr(board_mat)

    results = {
        "mat": mat,
        "chosen_map_keys": eval.chosen_map_keys,
        "board_mat": board_mat,
    }
    return results

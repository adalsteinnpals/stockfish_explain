import pytest
from stockfish import Stockfish
from utils import stockfish_eval
import yaml


def get_config():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = get_config()


def test_stockfish():

    sf = Stockfish(path=config["STOCKFISH_PATH"])
    print(f"Best move is: {sf.get_best_move()}")


def test_eval_stockfish():

    sfe = stockfish_eval(path=config["STOCKFISH_PATH"])
    sfe.set_fen_position("3q2k1/5b1p/1pr2pp1/p2p4/3P2PP/1P1NQP2/P2R2K1/8 w - - 0 1")
    df, eval_dict = sfe.get_evaluation_table(verbose=True)
    print(f"Evaluation is: {eval_dict}")


if __name__ == "__main__":
    test_stockfish()

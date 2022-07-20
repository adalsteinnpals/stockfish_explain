import pytest
from stockfish import Stockfish
import yaml


def get_config():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = get_config()


def test_stockfish():

    sf = Stockfish(path=config["STOCKFISH_PATH"])
    print(f"Best move is: {sf.get_best_move()}")


if __name__ == "__main__":
    test_stockfish()

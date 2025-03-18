from sys import argv
from os import environ
from ml import get_model_result
from ui.fen import FEN
from ui.board import draw
from engine import get_best_move

MODEL_PATH = environ.get("MODEL_PATH", "")
STOCKFISH_PATH = environ.get("STOCKFISH_PATH", "stockfish")


def main():
    chess_board = get_model_result(argv[1], MODEL_PATH)
    fen = FEN(chess_board).get_fen()
    move = get_best_move(fen, STOCKFISH_PATH)
    draw(fen, move)


if __name__ == "__main__":
    main()

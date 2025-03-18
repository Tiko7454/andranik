import torch
from .model.chess_piece_nn import ChessPieceNN


def load_model(path: str) -> ChessPieceNN:
    ckp = torch.load(path, map_location="cpu")
    model = ChessPieceNN()
    model.load_state_dict(ckp["model_state_dict"])
    model.eval()
    return model

from collections import defaultdict
import os
from typing import Optional
import cv2
import kagglehub
from tqdm import tqdm
from chess import Board, parse_square, Piece


def get_piece_notation(board, square_name):
    return piece_to_notation(board.piece_at(square_name))


def piece_to_notation(piece: Optional[Piece]) -> str:
    if piece is None:
        return "ee"
    piece_symbol = piece.symbol()
    color = "w" if piece_symbol.isupper() else "b"
    return f"{color}{piece_symbol.lower()}"


path = kagglehub.dataset_download("koryakinp/chess-positions")
print("Path to dataset files:", path)

directory = "train"
data_directory = "data"
for img_name in tqdm(
    os.listdir(os.path.join(path, directory))[:20000], desc="Processing"
):
    fen = img_name.split(".")[0].replace("-", "/")
    board = Board(fen)
    img_path = os.path.join(path, directory, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)
    height, width, _ = image.shape
    rows, cols = 8, 8
    square_height = height // rows
    square_width = width // cols

    # squares = {}
    result = defaultdict(list)
    for i in range(rows):
        for j in range(cols):
            x_start = j * square_width
            y_start = i * square_height
            square = cv2.resize(
                image[
                    y_start : y_start + square_height, x_start : x_start + square_width
                ],
                (48, 48),
            )
            col = chr(j + ord("a"))
            row = chr((7 - i) + ord("1"))
            square_name = parse_square(col + row)
            # squares[square_name] = square

            piece_notation = get_piece_notation(board, square_name)
            filename = str(abs(hash((fen, i, j))))
            filepath = f"{data_directory}/{piece_notation}/{filename}.png"
            cv2.imwrite(filepath, square)

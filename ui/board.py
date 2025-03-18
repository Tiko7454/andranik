from chess import parse_square
from fentoimage.board import BoardImage

def draw(fen, best_move):
    renderer = BoardImage(fen)
    sq1 = parse_square(best_move[:2])
    sq2 = parse_square(best_move[2:])
    image = renderer.render(highlighted_squares=(sq1, sq2))
    image.show()

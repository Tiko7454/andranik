from stockfish import Stockfish

def get_best_move(fen: str, stockfish_path: str):
    stockfish = Stockfish(path=stockfish_path)
    stockfish.set_depth(20)
    stockfish.set_skill_level(20)
    stockfish.set_fen_position(fen)
    return stockfish.get_best_move()
